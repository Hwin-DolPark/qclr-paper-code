import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_provider.uea import (
    normalize_batch_ts,
)
import warnings
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted

warnings.filterwarnings("ignore")


class DependentLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        # load data in subject-dependent manner
        self.X, self.y = self.load_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_dependent(self, data_path, label_path, flag=None):
        '''
        Loads data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        # print(filenames)
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        # 60 : 20 : 20
        X_train, y_train = np.array(feature_list), np.array(label_list)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train[:, 0]
        elif flag == 'VAL':
            return X_val, y_val[:, 0]
        elif flag == 'TEST':
            return X_test, y_test[:, 0]
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        data_list = np.load(self.label_path)

        all_ids = list(data_list[:, 1])  # id of all samples
        val_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC
        test_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_apava(self, data_path, label_path, flag=None):
        """
        Loads APAVA data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ASANLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        fold = args.seed
        with open(args.root_path + str(fold) + '.pkl', 'rb') as f:
            fold_data = pickle.load(f)
        X_train = fold_data['X_train']
        X_val = fold_data['X_val']
        X_test = fold_data['X_test']

        y_train = fold_data['y_train']
        y_val = fold_data['y_val']
        y_test = fold_data['y_test']

        seq_train = fold_data['seq_train']+1
        seq_val = fold_data['seq_valid']+1
        seq_test = fold_data['seq_test']+1


        if flag == "TRAIN":
            self.train_ids = np.arange(X_train.shape[0])
            self.X = X_train
            self.y = y_train
            max_len = X_train.shape[1]
            lengths = seq_train
            self.seq = (np.arange(max_len) < lengths[:, np.newaxis]).astype(int)
        elif flag == "VAL":
            self.val_ids = np.arange(X_val.shape[0])
            self.X = X_val
            self.y = y_val
            max_len = X_val.shape[1]
            lengths = seq_val
            self.seq = (np.arange(max_len) < lengths[:, np.newaxis]).astype(
                int)
        elif flag == "TEST":
            self.test_ids = np.arange(X_test.shape[0])
            self.X = X_test
            self.y = y_test
            max_len = X_test.shape[1]
            lengths = seq_test
            self.seq = (np.arange(max_len) < lengths[:, np.newaxis]).astype(
                int)

        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return (torch.from_numpy(self.X[index]),
                torch.from_numpy(np.asarray(self.y[index])),
                torch.from_numpy(np.asarray(self.seq[index])))

    def __len__(self):
        return len(self.y)

class MIMICLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        fold = args.seed
        with open(args.root_path + str(fold) + '.pkl', 'rb') as f:
            fold_data = pickle.load(f)
        X_train = fold_data['X_train']
        X_val = fold_data['X_val']
        X_test = fold_data['X_test']

        if args.model == 'Medformer':
            zeros = np.zeros((X_train.shape[0], 1, X_train.shape[2]), dtype=X_train.dtype)
            X_train = np.concatenate([X_train, zeros], axis=1)
            zeros = np.zeros((X_val.shape[0], 1, X_val.shape[2]),
                             dtype=X_val.dtype)
            X_val = np.concatenate([X_val, zeros], axis=1)
            zeros = np.zeros((X_test.shape[0], 1, X_test.shape[2]),
                             dtype=X_test.dtype)
            X_test = np.concatenate([X_test, zeros], axis=1)

        y_train = fold_data['y_train']
        y_val = fold_data['y_val']
        y_test = fold_data['y_test']

        seq_train = fold_data['seq_train']
        seq_val = fold_data['seq_valid']
        seq_test = fold_data['seq_test']

        if flag == "TRAIN":
            self.train_ids = np.arange(X_train.shape[0])
            self.X = X_train
            self.y = y_train
            max_len = X_train.shape[1]
            lengths = seq_train
            self.seq = (np.arange(max_len) < lengths[:, np.newaxis]).astype(int)
        elif flag == "VAL":
            self.val_ids = np.arange(X_val.shape[0])
            self.X = X_val
            self.y = y_val
            max_len = X_val.shape[1]
            lengths = seq_val
            self.seq = (np.arange(max_len) < lengths[:, np.newaxis]).astype(
                int)
        elif flag == "TEST":
            self.test_ids = np.arange(X_test.shape[0])
            self.X = X_test
            self.y = y_test
            max_len = X_test.shape[1]
            lengths = seq_test
            self.seq = (np.arange(max_len) < lengths[:, np.newaxis]).astype(
                int)

        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return (torch.from_numpy(self.X[index]),
                torch.from_numpy(np.asarray(self.y[index])),
                torch.from_numpy(np.asarray(self.seq[index])))

    def __len__(self):
        return len(self.y)

class PTBLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.55, 0.7

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptb(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        my_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial infarction IDs

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        """
        Loads ptb data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)
