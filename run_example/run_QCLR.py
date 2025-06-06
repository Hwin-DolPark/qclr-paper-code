import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        # required=True,
        default="classification_qclr",
        help="task name, options:[classification, classification_qclr]",
    )
    parser.add_argument(
        # "--is_training", type=int, required=True, default=1, help="status"
        "--is_training", type=int, default=1, help="status"
    )
    parser.add_argument(
        # "--model_id", type=str, required=True, default="PTB-Indep", help="model id"
        "--model_id", type=str, default="PTB-QCLR-Seed4145", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        # required=True,
        default="QCLR",
        help="model name, options: [Medformer, Transformer, iTransformer, PatchTST, MTST, QCLR]",
    )

    #QCLR loss configurations
    parser.add_argument('--q_lambda', type=float, default=0.1,
                        help='Weight lambda for QCLR loss')
    parser.add_argument('--upper_threshold', type=float, default=0.6,
                        help='Theta neg upper for QCLR loss')
    parser.add_argument('--init_lower_threshold', type=float, default=0.3,
                        help='Initial theta negative lower for QCLR loss')
    parser.add_argument('--lower_threshold', type=float, default=0.4,
                        help='Theta negative lower for QCLR loss')
    parser.add_argument('--positive_threshold', type=float, default=0.95,
                        help='Theta positive for QCLR loss')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                        help='epsilon for QCLR loss')

    #QCLR model configurations
    parser.add_argument('--sim_dim', type=int, default=96,
                        help='Similarity Projector - Output Dimension (v)')

    # data loader
    parser.add_argument(
        "--data", type=str, default="PTB", help="dataset type [PTB, APAVA, ASAN, MIMIC]"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/hpcmate/workspace/QCLR_Paper/data/PTB",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)"
    )

    # model define for baselines
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=6, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=256, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="whether to use inter-attention in encoder, using this argument means not using inter-attention",
        default=False,
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16, help="chunk_size used in LightTS"
    )
    parser.add_argument(
        "--patch_len", type=int, default=16, help="patch_len used in PatchTST"
    )
    parser.add_argument("--stride", type=int, default=8, help="stride used in PatchTST")
    parser.add_argument(
        "--sampling_rate", type=int, default=256, help="frequency sampling rate"
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="2,4,8,8,16,16,16,32,32,32,32,32",
        help="a list of patch len used in Medformer",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for Medformer",
        default=False,
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="none,drop0.5",
        help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )

    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=5, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=50, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="Exp", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging",
        default=False,
    )

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU 0 to use
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0", help="devi5e ids of multiple gpus"
    )

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    if args.task_name == "classification":
        Exp = Exp_Classification
    elif args.task_name == "classification_qclr":
        Exp = Exp_Classification
    else:
        print('Error: wrong task_name')

    if args.is_training:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            # setting record of experiments
            args.seed = seed
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.seed,
            )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            args.seed = seed
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.seed,
            )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
