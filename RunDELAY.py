import argparse
import os
import sys
import glob
import pytorch_lightning as pl
from tqdm import tqdm

from torch.utils.data                          import DataLoader
from torch.utils.data                          import ConcatDataset
from pytorch_lightning.loggers                 import TensorBoardLogger
from pytorch_lightning.callbacks               import ModelCheckpoint
from pytorch_lightning.plugins                 import DDPPlugin

from DELAY.Dataset import Dataset
from DELAY.Classifier import Classifier
from Networks.VGG_CNNC import VGG_CNNC
from Networks.SiameseVGG import SiameseVGG
from Networks.vgg import VGG

sys.path.append('Networks/')

if __name__ == '__main__':

    # ----------------
    # argument parser    TO-DO: ARGUMENT DESCRIPTIONS
    # ----------------
    parser = argparse.ArgumentParser(prog = 'DELAY', description = 'Depicting pseudotime-lagged causality for accurate gene-regulatory inference')
    parser.add_argument('datadir', help = 'Full path to directory containing one or more single-cell datasets')
    parser.add_argument('outdir', help = 'Relative directory for the logged results')
    parser.add_argument('-p', '--predict', action = 'store_true', help = 'Use a pre-trained model to predict interactions')
    parser.add_argument('-ft', '--finetune', action = 'store_true', help = 'Fine-tune a pre-trained model')
    parser.add_argument('-m', '--model', help = 'Full path to pre-trained model')
    parser.add_argument('-k', '--valsplit', type = int, help = '')
    parser.add_argument('-bs', '--batch_size', type = int, default = 512, help = '')
    parser.add_argument('-d', '--dimensions', type = int, dest = 'nbins', default = 32, help = '')
    parser.add_argument('-nb', '--neighbors', type = int, default = 2, help = '')
    parser.add_argument('-lg', '--max_lag', type = int, default = 5, help = '')
    parser.add_argument('-lr', '--learning_rate', type = float, default = .5, help = '')
    parser.add_argument('-e', '--max_epochs', type = int, default = 100, help = '')
    parser.add_argument('--train', action = 'store_true', help = 'Train a new model from scratch')
    parser.add_argument('--test', action = 'store_true', help = 'Test a pre-trained model')
    parser.add_argument('-cfg', '--model_cfg', nargs = '*', default = ['1024', 'M', '512', 'M', '256', 'M', '128', 'M', '64'], help = '')
    parser.add_argument('--model_type', choices = ['inverted-vgg', 'vgg-cnnc', 'siamese-vgg', 'vgg'], default = 'inverted-vgg', help = '')
    parser.add_argument('--mask_lags', type = int, nargs = '*',  default = [], help = '')
    parser.add_argument('--mask_region', choices = ['off-off', 'on-off', 'off-on', 'on-on', 'on'], help = '')
    parser.add_argument('--shuffle_traj', type = float, dest = 'shuffle', help = '')
    parser.add_argument('--ncells_traj', type = int, dest = 'ncells', help = '')
    parser.add_argument('--dropout_traj', type = float, dest = 'dropout', help = '')
    parser.add_argument('--auc_motif', dest = 'motif', choices = ['ffl-reg', 'ffl-tgt', 'ffl-trans', 'fbl-trans', 'mi-simple'], help = '')
    parser.add_argument('--ablate_genes', dest = 'ablate', action = 'store_true', help = '')
    parser.add_argument('--check_val_every_n_epoch', type = int, default = 1, help = '')
    parser.add_argument('--workers', type = int, default = 36, help = '')
    parser.add_argument('--gpus', type = int, default = 2, help = '')
    args = parser.parse_args()

    # ------------------
    # set up run for pl
    # ------------------
    pl.seed_everything(1234); callbacks = None
    if args.train == True: prefix = 'val_'
    elif args.test == True: prefix = 'test_'
    elif args.predict == True: prefix = 'pred_'

    # ---------------------------------------------
    # load/compile mini-batches for given datasets
    # ---------------------------------------------
    print('Loading datasets...')
    training, validation, valnames = [], [], []
    for f in tqdm(sorted(glob.glob(f'{args.datadir}*/'))):
        if os.path.isdir(f):

            # training/validation dsets (training/fine-tuning)
            train_ds, val_ds = None, None
            if args.train == True or args.finetune == True:
                train_ds = Dataset(args, f, 'training')
                if args.valsplit is not None:
                    val_ds = Dataset(args, f, 'validation')

            # testing/prediction dsets - ONLY (no fine-tuning)
            elif args.test == True:
                val_ds = Dataset(args, f, 'validation')
            elif args.predict == True:
                train_ds = Dataset(args, f, 'prediction')

            # update lists, names for dsets
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
        training = ConcatDataset(training)
        train_loader = DataLoader(training, batch_size = None, shuffle = True, num_workers = args.workers, pin_memory = True)
        
    if len(validation) > 0:
        val_loader = [None] * len(validation)
        for i in range(len(validation)):
            val_loader[i] = DataLoader(validation[i], batch_size = None, num_workers = args.workers, pin_memory = True)

    # ----------------------------------------------------------
    # neural-network backbone with specified type/configuration
    # ----------------------------------------------------------
    args.model_cfg = [int(x) if x not in ['M','D'] else x for x in args.model_cfg]
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





    ## start here

    logger = TensorBoardLogger('lightning_logs', name = args.outdir)

    # ----------
    # callbacks
    # ----------
    if args.train == True or args.finetune == True:
        if args.train_split < 1.:
            ckpt_fn = '{epoch}-{' + prefix + 'avg_auprc:.3f}-{' + prefix + 'avg_auroc:.3f}'
            monitor, mode = f'{prefix}avg_auc', 'max'
        else:
            monitor, mode, ckpt_fn = 'train_loss', 'min', '{epoch}-{train_loss:.6f}'

        callbacks = [ModelCheckpoint(monitor = monitor, mode = mode, save_top_k = 1,
                     dirpath = f'lightning_logs/{args.outdir}/', filename = ckpt_fn)]

    # -----------
    # pl trainer
    # -----------
    trainer = pl.Trainer(max_epochs = args.max_epochs, deterministic = True,
                         accelerator = 'ddp', gpus = args.gpus, auto_select_gpus = True,
                         logger = logger, callbacks = callbacks, num_sanity_val_steps = 0,
                         plugins = [ DDPPlugin(find_unused_parameters=False) ],
                         check_val_every_n_epoch = args.check_val_every_n_epoch)

    # ------------------
    # train (from init)
    # ------------------
    if args.train == True:
        trainer.fit(model, train_loader, val_loader)

    # ------------------------
    # test (from pre-trained)
    # ------------------------
    elif args.test == True:
        trainer.test(model, val_loader)

        # ---------------------------
        # finetune (with validation)
        # ---------------------------
        if args.finetune == True:
            trainer.fit(model, train_loader, val_loader)

    # ---------------------------
    # predict (from pre-trained)
    # ---------------------------
    elif args.predict == True:

        # --------------------------
        # finetune (opt validation)
        # --------------------------
        if args.finetune == True:
            if args.train_split == 1.: trainer.fit(model, train_loader)
            else: trainer.fit(model, train_loader, val_loader)

        # ---------------------------
        # evaluation (no validation)
        # ---------------------------
        else: trainer.predict(model, train_loader)