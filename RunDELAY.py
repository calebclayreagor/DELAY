import argparse
import os
import sys
import glob
import pytorch_lightning as pl

from torch.utils.data                          import DataLoader
from torch.utils.data                          import ConcatDataset
from pytorch_lightning.loggers                 import TensorBoardLogger
from pytorch_lightning.callbacks               import ModelCheckpoint
from pytorch_lightning.callbacks               import LearningRateMonitor

from DELAY.Dataset import Dataset
from DELAY.Classifier import Classifier
from Networks.VGG_CNNC import VGG_CNNC
from Networks.SiameseVGG import SiameseVGG
from Networks.vgg import VGG
from Networks.GCN import GCN

import numpy as np

sys.path.append('Networks/')

if __name__ == '__main__':

    # ----------------
    # argument parser
    # ----------------
    parser = argparse.ArgumentParser(prog = 'DELAY', description = 'DELAY: DEpicting pseudotime-LAgged causalitY across single-cell trajectories for accurate gene-regulatory inference')
    parser.add_argument('datadir', help = 'full path to directory with >=1 sub-directory with required input files')
    parser.add_argument('outdir', help = 'relative path for logged results/hyperparameters and saved model checkpoints')
    parser.add_argument('-p', '--predict', action = 'store_true', help = 'predict gene-regulation probabilities using pre-trained model')
    parser.add_argument('-ft', '--finetune', action = 'store_true', help = 'fine-tune model with partially-known ground truths (e.g. from ChIP-seq)')
    parser.add_argument('-m', '--model', metavar = 'CKPT_FILE', help = 'full path to saved checkpoint file with pre-trained model weights')
    parser.add_argument('-k', '--val_fold', metavar = 'K', dest = 'valsplit', type = int, help = 'data fold/split to hold out for validation (optional)')
    parser.add_argument('-bs', '--batch_size', metavar = 'BS', type = int, default = 32, help = 'number of TF-target examples per mini-batch')
    parser.add_argument('-d', '--dimensions', metavar = 'D', dest = 'nbins', type = int, default = 32, help = 'number of gene-expression levels used to bin data for input matrices')
    parser.add_argument('-nb', '--neighbors', metavar = 'NB', type = int, default = 2, help = 'number of neighbors used for gene pairs in input')
    parser.add_argument('-lg', '--max_lag', metavar = 'LG', type = int, default = 5, help = 'number of lagged matrices to include in input')
    parser.add_argument('-lr', '--learning_rate', metavar = 'LR', type = float, default = .5)
    parser.add_argument('-e', '--training_epochs', metavar = 'E', type = int, default = 100)
    parser.add_argument('-w', '--workers', metavar = 'W', type = int, default = os.cpu_count(), help = 'number of sub-processes for mini-batch loading')
    parser.add_argument('-g', '--gpus', metavar = 'G', type = int, default = -1, help = 'number of GPUs for distributed training')
    parser.add_argument('--train', action = 'store_true', help = 'train new model from scratch')
    parser.add_argument('--test', action = 'store_true', help = 'test pre-trained model on augmented data/inputs')
    parser.add_argument('-cfg', '--model_config', metavar = 'LAYER', dest = 'model_cfg', nargs = '*', default = ['1024', 'M', '512', 'M', '256', 'M', '128', 'M', '64'], 
        help = 'configure convolutional and max-pooling layers for feature extractor (e.g. 32 32 M 64 64 M ...)')
    parser.add_argument('--model_type', choices = ['inverted-vgg', 'vgg-cnnc', 'siamese-vgg', 'vgg', 'gcn'], default = 'inverted-vgg')
    parser.add_argument('--mask_lags', metavar =  'LG', type = int, nargs = '*',  default = [], help = 'mask inputs at specified lags')
    parser.add_argument('--mask_region', choices = ['off-off', 'on-off', 'off-on', 'on-on', 'on'], help = 'mask regions of input matrices')
    parser.add_argument('--shuffle_trajectory', metavar = 'FRAC', dest = 'shuffle', type = float, help = 'shuffle cells in local windows across trajectory')
    parser.add_argument('--ncells_trajectory', metavar = 'N', dest = 'ncells', type = int, help = 'randomly sample cells from trajectory')
    parser.add_argument('--dropout_trajectory', metavar = 'P', dest = 'dropout', type = float, help = 'include additional sequencing dropouts with specified probability')
    parser.add_argument('--auc_motif', dest = 'motif', choices = ['ffl-reg', 'ffl-tgt', 'ffl-trans', 'fbl-trans', 'mi-simple'], help = 'compute AUC for examples in specified motif')
    parser.add_argument('--ablate_genes', dest = 'ablate', action = 'store_true', help = 'mask input matrices for neighbors in specified motif')
    parser.add_argument('--graphs') # path to graphs directory
    args = parser.parse_args()

    # ---------------------------------
    # set up run for pytorch_lightning
    # ---------------------------------
    callback, loss_freq = None, 1
    pl.seed_everything(1234)
    if args.train == True: prefix = 'val_'
    elif args.test == True: prefix = 'test_'
    elif args.predict == True: prefix = 'pred_'

    # ------------------------------------------------
    # load/compile mini-batches for provided datasets
    # ------------------------------------------------
    print('Loading datasets...')
    training, validation, valnames = [], [], []
    for f in sorted(glob.glob(f'{args.datadir}*/')):
        if os.path.isdir(f):

            # training/validation dsets (training/fine-tuning)
            train_ds, val_ds = None, None
            if args.train == True or args.finetune == True:
                train_ds = Dataset(args, f, 'training')
                shuffle_train = True
                if args.valsplit is not None:
                    val_ds = Dataset(args, f, 'validation')

            # testing/prediction dsets [ONLY] (no fine-tuning)
            elif args.test == True:
                val_ds = Dataset(args, f, 'validation')
            elif args.predict == True:
                train_ds = Dataset(args, f, 'prediction')
                shuffle_train = False

            # update dsets lists/names
            if train_ds is not None: 
                training.append(train_ds)
            if val_ds is not None:
                validation.append(val_ds)
                name = '_'.join(f.split('/')[-2:])
                valnames.append(prefix + name)

    # --------------------------------
    # training/validation dataloaders
    # --------------------------------
    if len(training) > 0:
        training = ConcatDataset(training)  # training dataloader is also used for prediction
        train_loader = DataLoader(training, batch_size = None, shuffle = shuffle_train, num_workers = args.workers, pin_memory = True)
        loss_freq = int(round(len(train_loader) / 50) + 1)

    if len(validation) > 0:
        val_loader = [None] * len(validation)
        for i in range(len(validation)):  # validation dataloader is also used for testing (no fine-tuning)
            val_loader[i] = DataLoader(validation[i], batch_size = None, num_workers = args.workers, pin_memory = True)

    # --------------------------------------------------------
    # NN backbone with specified model_type and configuration
    # --------------------------------------------------------
    args.model_cfg = [int(x) if x != 'M' else x for x in args.model_cfg]
    nchan = (3 + 2 * args.neighbors) * (1 + args.max_lag)
    if args.model_type == 'inverted-vgg': net = VGG(cfg = args.model_cfg, in_channels = nchan)
    elif args.model_type == 'vgg-cnnc': net = VGG_CNNC(cfg = args.model_cfg, in_channels = 1)
    elif args.model_type == 'siamese-vgg': net = SiameseVGG(cfg = args.model_cfg, neighbors = args.neighbors, max_lag = args.max_lag)
    elif args.model_type == 'vgg': net = VGG_CNNC(cfg = args.model_cfg, in_channels = nchan)
    elif args.model_type == 'gcn': net = GCN(graphs = args.graphs, cfg = args.model_cfg, in_dimensions = args.nbins * (2 + args.neighbors))

    # ---------------------------------------------------------
    # set up classifier from scratch or pre-trained checkpoint
    # ---------------------------------------------------------
    if args.train == True: model = Classifier(args, net, valnames, prefix)
    else: model = Classifier.load_from_checkpoint(args.model, hparams = args, backbone = net, valnames = valnames, prefix = prefix)

    # --------------------------------------------------
    # set up callback and trainer for pytorch_lightning
    # --------------------------------------------------
    if args.train == True or args.finetune == True:
        if args.valsplit is not None: monitor, mode, fn = f'{prefix}avg_auc', 'max', f"{'BEST_WEIGHTS_{'}{prefix}{'avg_auc:.3f}_{epoch}'}"
        else: monitor, mode, fn = 'train_loss', 'min', 'BEST_WEIGHTS_{train_loss:.3f}_{epoch}'
        callback = ModelCheckpoint(monitor = monitor, mode = mode, filename = fn, save_top_k = 1, dirpath = f'RESULTS/{args.outdir}/')

    trainer = pl.Trainer(strategy = 'ddp_find_unused_parameters_false', accelerator = 'gpu', devices = args.gpus, auto_select_gpus = True, 
                         max_epochs = args.training_epochs, num_sanity_val_steps = 0, log_every_n_steps = loss_freq,
                         deterministic = 'warn', callbacks = callback, logger = TensorBoardLogger('RESULTS', name = args.outdir))

    # -------------------------
    # train model from scratch
    # -------------------------
    if args.train == True:
        trainer.fit(model, train_loader, val_loader)

    # --------------------------------------------------------------
    # test/predict from pre-trained model with optional fine-tuning
    # --------------------------------------------------------------
    elif args.test == True:
        trainer.test(model, val_loader)
        if args.finetune == True:  # with validation
            trainer.fit(model, train_loader, val_loader)

    elif args.predict == True:
        if args.finetune == True:  # with optional validation
            if args.valsplit is not None:
                trainer.fit(model, train_loader, val_loader)
            else: trainer.fit(model, train_loader)
        else: trainer.predict(model, train_loader)