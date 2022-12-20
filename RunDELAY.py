import argparse
import os
import sys
import glob
import pytorch_lightning as pl

from torch.utils.data                          import DataLoader
from torch.utils.data                          import ConcatDataset
from pytorch_lightning.loggers                 import TensorBoardLogger
from pytorch_lightning.callbacks               import ModelCheckpoint

from DELAY.Dataset import Dataset
from DELAY.Classifier import Classifier
from Networks.VGG_CNNC import VGG_CNNC
from Networks.SiameseVGG import SiameseVGG
from Networks.vgg import VGG

sys.path.append('Networks/')

if __name__ == '__main__':

    # ----------------
    # argument parser
    # ----------------
    parser = argparse.ArgumentParser(prog = 'DELAY', description = 'Depicting pseudotime-lagged causality for accurate gene-regulatory inference')
    parser.add_argument('datadir', help = 'Full path to directory containing one or more single-cell datasets for model training, testing, or prediction')
    parser.add_argument('outdir', help = 'Relative path for logged results, hyperparameters and checkpoint files containing the best model weights')
    parser.add_argument('-p', '--predict', action = 'store_true', help = 'Predict gene-regulation probabilities for TF-target gene pairs using a pre-trained model')
    parser.add_argument('-ft', '--finetune', action = 'store_true', help = 'Fine-tune a pre-trained model on partially-known ground-truth interactions')
    parser.add_argument('-m', '--model', help = 'Full path to saved checkpoint file containing best weights for a pre-trained model')
    parser.add_argument('-k', '--valsplit', type = int, help = 'Optional label of data fold/split to hold out for model validation')
    parser.add_argument('-bs', '--batch_size', type = int, default = 512, help = 'Number of TF-target gene-pair examples per mini-batch')
    parser.add_argument('-d', '--dimensions', type = int, dest = 'nbins', default = 32, help = 'Number of gene-expression levels used in data binnning to generate joint-probability matrices')
    parser.add_argument('-nb', '--neighbors', type = int, default = 2, help = 'Number of highly-correlated neighbor genes to include as input with each TF-target gene-pair example')
    parser.add_argument('-lg', '--max_lag', type = int, default = 5, help = 'Number of pseudotime-lagged input matrices to include with each TF-target gene-pair example')
    parser.add_argument('-lr', '--learning_rate', type = float, default = .5, help = 'Learning rate to use for model training or fine-tuning')
    parser.add_argument('-e', '--max_epochs', type = int, default = 100, help = 'Number of epochs to use for model training or fine-tuning')
    parser.add_argument('-ve', '--valfreq', type = int, default = 1, help = 'Option to skip validation for the specified number of epochs')
    parser.add_argument('-w', '--workers', type = int, default = 2, help = 'Number of workers (sub-processes) to use for loading mini-batches')
    parser.add_argument('-g', '--gpus', type = int, default = -1, help = 'Number of automatically-selected GPUs to use for distributed training')
    parser.add_argument('--train', action = 'store_true', help = 'Train a new model from scratch')
    parser.add_argument('--test', action = 'store_true', help = 'Test a pre-trained model, often using augmented datasets or input matrices')
    parser.add_argument('-cfg', '--model_cfg', nargs = '*', default = ['1024', 'M', '512', 'M', '256', 'M', '128', 'M', '64'], help = ('Configuration of feature extractor(s) for the network,', 
                        'including max-pooling and convolutional layers with specified numbers of activation maps'))
    parser.add_argument('--model_type', choices = ['inverted-vgg', 'vgg-cnnc', 'siamese-vgg', 'vgg'], default = 'inverted-vgg', help = 'Choice of overall network architecture for the model')
    parser.add_argument('--mask_lags', type = int, nargs = '*',  default = [], help = 'Option to mask the input matrices at the specified pseudotime lags')
    parser.add_argument('--mask_region', choices = ['off-off', 'on-off', 'off-on', 'on-on', 'on'], help = 'Option to mask the specified region(s) of the input matrices')
    parser.add_argument('--shuffle_traj', type = float, dest = 'shuffle', help = 'Option to shuffle cells along trajectories at given length scales (fraction of trajectory from 0 to 1)')
    parser.add_argument('--ncells_traj', type = int, dest = 'ncells', help = 'Option to randomly sample the specified number of cells from each trajectory')
    parser.add_argument('--dropout_traj', type = float, dest = 'dropout', help = 'Option to include additional sequencing dropouts at the specified rate (fraction from 0 to 1)')
    parser.add_argument('--auc_motif', dest = 'motif', choices = ['ffl-reg', 'ffl-tgt', 'ffl-trans', 'fbl-trans', 'mi-simple'],
                        help = 'Option to compute AUC values for only the gene pairs in a specified regulatory motif')
    parser.add_argument('--ablate_genes', dest = 'ablate', action = 'store_true', help = 'Option to mask neighbor genes participating in the specified regulatory motif')
    args = parser.parse_args()

    # ---------------------------------
    # set up run for pytorch_lightning
    # ---------------------------------
    pl.seed_everything(1234); callback = None
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
                if args.valsplit is not None:
                    val_ds = Dataset(args, f, 'validation')

            # testing/prediction dsets [ONLY] (no fine-tuning)
            elif args.test == True:
                val_ds = Dataset(args, f, 'validation')
            elif args.predict == True:
                train_ds = Dataset(args, f, 'prediction')

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
        train_loader = DataLoader(training, batch_size = None, shuffle = True, num_workers = args.workers, pin_memory = True)
        
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
    elif args.model_type == 'siamese-vgg': net = SiameseVGG(cfg = args.model_cfg, neighbors = args.neighbors)
    elif args.model_type == 'vgg': net = VGG_CNNC(cfg = args.model_cfg, in_channels = nchan)

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
                         max_epochs = args.max_epochs, num_sanity_val_steps = 0, check_val_every_n_epoch = args.valfreq,
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