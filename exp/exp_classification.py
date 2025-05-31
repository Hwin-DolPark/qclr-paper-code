from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import math
import time
import warnings
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa
        self.train_epochs = args.train_epochs

    def qclr_loss_dynamic(self, latent_vector, true_y, current_epoch,
                          upper_threshold=0.6,  # Upper quantile for negatives
                          # Lower quantile for negatives
                          initial_lower_threshold=0.3,
                          target_lower_threshold=0.4,
                          # Quantile to define positives
                          positive_quantile=0.95,
                          temperature=0.1,
                          total_epochs=50,
                          epsilon=1e-6):  # Small value for numerical stability
        """
        Computes a contrastive loss focusing on intra-class similarity.

        Pulls together high-similarity pairs within the same class (positives)
        and pushes apart lower-similarity pairs within the same class (negatives).
        Does not contrast across different classes.
        """
        total_epochs = self.train_epochs
        # --- Parameters for the FIXED Threshold Ablation Run ---
        batch_size = latent_vector.size(0)
        device = latent_vector.device

        # 1. Normalize latent vectors
        latent_vector = F.normalize(latent_vector, dim=1)

        # 2. Anneal the lower threshold for defining negatives
        # (Using exponential decay as before)
        if total_epochs > 0 and current_epoch < total_epochs:
            # Prevent division by zero or log(0) if epsilon is too small / total_epochs large
            # Adjust k_lower calculation if needed, this assumes epsilon is reachable
            k_lower_arg = -math.log(max(epsilon, 1e-9)) / total_epochs
            k_lower = max(0, k_lower_arg)  # Ensure k_lower is non-negative

            annealed_lower_threshold = target_lower_threshold + (
                    (
                                initial_lower_threshold - target_lower_threshold) * math.exp(
                -k_lower * current_epoch
            )
            )
            # Clamp within initial and target bounds
            lower_threshold = max(initial_lower_threshold,
                                  min(target_lower_threshold,
                                      annealed_lower_threshold)) \
                if initial_lower_threshold < target_lower_threshold \
                else min(initial_lower_threshold,
                         max(target_lower_threshold, annealed_lower_threshold))
        else:
            lower_threshold = target_lower_threshold  # Use target if epochs are done or invalid

        # Ensure thresholds are valid quantiles
        lower_threshold = max(0.0, min(1.0, lower_threshold))
        upper_threshold = max(0.0, min(1.0, upper_threshold))
        positive_quantile = max(0.0, min(1.0, positive_quantile))
        if lower_threshold >= upper_threshold or lower_threshold >= positive_quantile:
            print(
                f"Warning: Quantile thresholds overlap improperly. Lower: {lower_threshold:.3f}, Upper: {upper_threshold:.3f}, Positive: {positive_quantile:.3f}")

        # 3. Calculate pairwise cosine similarity
        similarity_matrix = torch.matmul(latent_vector, latent_vector.T)

        # 4. Create masks
        labels = true_y.contiguous().view(-1, 1)
        mask_same_label = torch.eq(labels, labels.T).float()
        # Mask out self-comparisons
        mask_diag_off = (1.0 - torch.eye(batch_size, device=device)).float()
        mask_same_label_no_diag = mask_same_label * mask_diag_off

        # --- Calculate Quantiles Per Anchor (Row-wise) ---
        # We only want quantiles based on *valid same-label* pairs for each anchor
        relevant_similarities = similarity_matrix.clone()
        # Fill values we DON'T want to consider for quantile with a value outside [-1, 1] range, like -2
        relevant_similarities.masked_fill_(mask_same_label_no_diag == 0, -2.0)

        # Calculate row-wise quantiles, ignoring the -2.0 values
        # This requires a loop or more complex vectorized approach if torch.quantile doesn't handle NaNs well
        # Let's loop for clarity first, then consider vectorization improvements if needed.
        positive_thresh_per_anchor = torch.full((batch_size, 1), -1.0,
                                                device=device)  # Initialize
        lower_thresh_per_anchor = torch.full((batch_size, 1), -1.0,
                                             device=device)
        upper_thresh_per_anchor = torch.full((batch_size, 1), -1.0,
                                             device=device)

        for i in range(batch_size):
            anchor_relevant_sims = relevant_similarities[i][
                mask_same_label_no_diag[i] == 1]
            if anchor_relevant_sims.numel() > 0:
                positive_thresh_per_anchor[i] = torch.quantile(
                    anchor_relevant_sims, positive_quantile)
                lower_thresh_per_anchor[i] = torch.quantile(
                    anchor_relevant_sims, lower_threshold)
                upper_thresh_per_anchor[i] = torch.quantile(
                    anchor_relevant_sims, upper_threshold)
            # else: keep -1.0, these anchors won't have pairs

        # --- Define Positive and Negative Pairs Based on Quantiles ---
        # Positive pairs: same label, no diag, AND sim >= positive_thresh_per_anchor
        mask_positives = (
                (
                            similarity_matrix >= positive_thresh_per_anchor)  # Broadcasting (bs, bs) >= (bs, 1)
                & (mask_same_label_no_diag == 1)
        ).float()

        # Negative pairs: same label, no diag, AND lower_thresh <= sim < upper_thresh
        # Note: Using '< upper_thresh' to avoid potential overlap with positives if quantiles are very close
        mask_negatives = (
                (similarity_matrix >= lower_thresh_per_anchor)
                & (
                            similarity_matrix < upper_thresh_per_anchor)  # Strict inequality recommended here
                & (mask_same_label_no_diag == 1)
        ).float()

        # 5. Calculate InfoNCE-style Loss per anchor
        # For each anchor `i`, we contrast each of its positives `p` against all of its negatives `n`

        # Apply temperature
        logits = similarity_matrix / temperature
        exp_logits = torch.exp(logits)

        # Calculate sum of exp(negatives) per anchor
        # Mask out non-negatives, then sum row-wise
        exp_negatives = exp_logits * mask_negatives
        sum_exp_negatives_per_anchor = exp_negatives.sum(dim=1,
                                                         keepdim=True)  # Shape: (batch_size, 1)

        # Calculate loss terms: log [ exp(pos) / (exp(pos) + sum_exp_neg) ]
        # Numerator: exp(positive logits)
        exp_positives = exp_logits * mask_positives  # Shape: (batch_size, batch_size), zero where not positive

        # Denominator: exp(positive logits) + sum_exp_negatives_per_anchor (broadcasted)
        denominator = exp_positives + sum_exp_negatives_per_anchor  # Broadcasting works here

        # Calculate log probabilities, masking out terms where denominator is zero or positive is zero
        log_probs = torch.log(exp_positives / (
                    denominator + epsilon) + epsilon)  # Add epsilon for stability

        # Mask out entries that are not positives
        log_probs_masked = log_probs * mask_positives

        # Calculate mean log-probability per anchor (average over its positive pairs)
        # Sum the log_probs for each anchor's positives
        sum_log_probs_per_anchor = log_probs_masked.sum(
            dim=1)  # Shape: (batch_size)
        # Count the number of positive pairs for each anchor
        num_positives_per_anchor = mask_positives.sum(
            dim=1)  # Shape: (batch_size)

        # Avoid division by zero for anchors with no positive pairs
        valid_anchor_mask = (num_positives_per_anchor > 0)
        # Also ensure anchors have negatives to contrast against
        valid_anchor_mask = valid_anchor_mask & (mask_negatives.sum(dim=1) > 0)

        # Calculate mean loss only for valid anchors
        loss_per_anchor = -sum_log_probs_per_anchor[valid_anchor_mask] / (
                    num_positives_per_anchor[valid_anchor_mask] + epsilon)

        # Final loss: mean over valid anchors in the batch
        if loss_per_anchor.numel() > 0:
            loss = loss_per_anchor.mean()
        else:
            # Handle case where no anchor had valid positive/negative pairs in this batch
            loss = torch.tensor(0.0, device=device,
                                requires_grad=True)  # Or False if loss=0 desired

        return loss

    def _build_model(self):
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(test_data.y))
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                elif self.args.task_name == 'classification_qclr':
                    outputs, _ = self.model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.args.task_name == 'classification_qclr':
                    outputs = self.model(batch_x, padding_mask, None, None)
                    output, output_qclr = outputs
                    loss1 = criterion(output, label.long())
                    # loss1 = criterion(output, label.long())*0.9
                    loss2 = self.qclr_loss_dynamic(output_qclr, label.long(), epoch)*0.1
                    # print(loss1, loss2)
                    loss = loss1 + loss2
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return
