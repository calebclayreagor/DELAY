import argparse
import os, sys
import glob
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from torch.utils.data                          import DataLoader
from torch.utils.data                          import random_split
from torch.utils.data                          import ConcatDataset
from pytorch_lightning.loggers                 import TensorBoardLogger
from pytorch_lightning.callbacks               import LearningRateMonitor
from pytorch_lightning.callbacks               import ModelCheckpoint
from pytorch_lightning.plugins                 import DDPPlugin

from DELAY.Dataset import Dataset
from DELAY.Classifier import Classifier
from Networks.VGG_CNNC import VGG_CNNC
from Networks.SiameseVGG import SiameseVGG
from Networks.vgg import VGG
sys.path.append('Networks/')

if __name__ == '__main__':

    # ----------
    # arguments
    # ----------
    parser = argparse.ArgumentParser(prog = 'DELAY', description = 'Depicting pseudotime-lagged causality for accurate gene-regulatory inference')
    parser.add_argument('datasets', help = 'Directory containing one or more single-cell datasets')
    parser.add_argument('output', help = '')
    parser.add_argument('--compile', action = 'store_true', help = 'Compile mini-batches of input matrices')
    parser.add_argument('--atac', action = 'store_true', help = 'Specify chromatin-accessibility datasets')
    parser.add_argument('--train', action = 'store_true', help = 'Train a new model from scratch')
    parser.add_argument('--test', action = 'store_true', help = 'Test a pre-trained model')
    parser.add_argument('-p', '--predict', action = 'store_true', help = 'Use a pre-trained model to predict interactions')
    parser.add_argument('-ft', '--finetune', action = 'store_true', help = 'Fine-tune a pre-trained model')

    parser.add_argument('--model_dir', type=str, default='') ##

    parser.add_argument('--split', type = float, nargs = '*', help = '')
    parser.add_argument('--validate', type = int, help = '')


    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--neighbors', type=int, default=2)
    parser.add_argument('--max_lag', type=int, default=5)

    parser.add_argument('--mask_lags', type = int, nargs = '*', help = '') # update code to deal with default (None)

    parser.add_argument('--nbins_img', type=int, default=32)
    parser.add_argument('--mask_region', type=str, default='')
    parser.add_argument('--shuffle_traj', type=float, default=0.)
    parser.add_argument('--ncells_traj', type=int, default=0)
    parser.add_argument('--dropout_traj', type=float, default=0.)
    parser.add_argument('--auc_motif', type=str, default='none')
    parser.add_argument('--ablate_genes', action = 'store_true')
    parser.add_argument('--lr_init', type=float, default=.5)
    parser.add_argument('--nn_dropout', type=float, default=0.)
    parser.add_argument('--model_cfg', type=str, default='')
    parser.add_argument('--model_type', type=str, default='inverted-vgg')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=36)
    parser.add_argument('--num_gpus', type=int, default=2)
    args = parser.parse_args()

    prefix, callbacks, cfg = '', None, []

    pl.seed_everything(1234)

    ## -------------------------
    ## train_split (evaluation)
    ## -------------------------
    #if args.predict == True and args.finetune == False: args.train_split = 1.

    # ------------------
    # data type (fname)
    # ------------------
    if args.atac == True:
        data_fname = 'AccessibilityData.csv'
    else: data_fname = 'ExpressionData.csv'

    # -------
    # prefix
    # -------
    if args.train == True: prefix = 'val_'
    elif args.test == True: prefix = 'test_'
    elif args.predict == True: prefix = 'pred_'

    # ---------------------------
    # datasets (train/val split)
    # ---------------------------
    print('Loading datasets...')
    training, validation, val_names = [], [], []
    for item in tqdm(sorted(glob.glob(args.datasets))):
        if os.path.isdir(item):

            dset = Dataset(root_dir=item,
                           rel_path=f'*/{data_fname}',
                           neighbors=args.neighbors,
                           max_lag=args.max_lag,
                           mask_lags=args.mask_lags,
                           nbins=args.nbins_img,
                           mask_img=args.mask_region,
                           shuffle=args.shuffle_traj,
                           ncells=args.ncells_traj,
                           dropout=args.dropout_traj,
                           motif=args.auc_motif,
                           ablate=args.ablate_genes,
                           tf_ref=args.predict,
                           use_tf=(not args.finetune),
                           batchSize=args.batch_size,
                           load_prev=(not args.compile))

            # ----------------
            # train/val split
            # ----------------
            if args.train_split < 1.0:
                train_size = int(args.train_split * len(dset))
                train_dset, val_dset = random_split(dset, [train_size, len(dset)-train_size],
                                       generator=torch.Generator().manual_seed(args.global_seed))
                val_names.append(prefix+'_'.join(item.split('/')[-2:]))
                training.append(train_dset)
                validation.append(val_dset)
            else:
                training.append(dset)

    training = ConcatDataset(training)
    train_loader = DataLoader(training, batch_size=None, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = [None] * len(validation)
    for i in range(len(validation)):
        val_loader[i] = DataLoader(validation[i], batch_size=None, num_workers=args.num_workers, pin_memory=True)

    if args.compile == True: 
        sys.exit("Successfuly compiled datasets.")

    # ---------
    # backbone
    # ---------
    for item in args.model_cfg.split(','):
        if item=='M': cfg.append('M')
        elif item=='D': cfg.append('D')
        else: cfg.append(int(item))
    args.model_cfg, nchans = cfg, (3+2*args.neighbors)*(1+args.max_lag)
    if args.model_type=='inverted-vgg':
        backbone = VGG(cfg=args.model_cfg, in_channels=nchans, dropout=args.nn_dropout)
    elif args.model_type=='vgg-cnnc':
        backbone = VGG_CNNC(cfg=args.model_cfg, in_channels=1, dropout=args.nn_dropout)
    elif args.model_type=='siamese-vgg':
        backbone = SiameseVGG(cfg=args.model_cfg, neighbors=args.neighbors, dropout=args.nn_dropout)
    elif args.model_type=='vgg':
        backbone = VGG_CNNC(cfg=args.model_cfg, in_channels=nchans, dropout=args.nn_dropout)

    # ----------------------------
    # model (init or pre-trained)
    # ----------------------------
    if args.train == True:
        model = Classifier(args, backbone, val_names, prefix)
    else:
        model = Classifier.load_from_checkpoint(args.model_dir, hparams=args,
                backbone=backbone, val_names=val_names, prefix=prefix)

    # -------
    # logger
    # -------
    logger = TensorBoardLogger('lightning_logs', name = args.output)

    # ----------
    # callbacks
    # ----------
    if args.train == True or args.finetune == True:
        if args.train_split < 1.:
            ckpt_fname = '{epoch}-{'+prefix+'avg_auprc:.3f}-{'+prefix+'avg_auroc:.3f}'
            monitor, mode = f'{prefix}avg_auc', 'max'
        else:
            monitor, mode, ckpt_fname = 'train_loss', 'min', '{epoch}-{train_loss:.6f}'

        callbacks = [ LearningRateMonitor(logging_interval='epoch'), ModelCheckpoint(
                      monitor=monitor, mode=mode, save_top_k=1,
                      dirpath=f"lightning_logs/{args.output}/", filename = ckpt_fname) ]

    # -----------
    # pl trainer
    # -----------
    trainer = pl.Trainer(max_epochs=args.max_epochs, deterministic=True,
                         accelerator='ddp', gpus=args.num_gpus, auto_select_gpus=True,
                         logger=logger, callbacks=callbacks, num_sanity_val_steps=0,
                         plugins=[ DDPPlugin(find_unused_parameters=False) ],
                         check_val_every_n_epoch=args.check_val_every_n_epoch)

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
            if args.train_split==1.: trainer.fit(model, train_loader)
            else: trainer.fit(model, train_loader, val_loader)

        # ---------------------------
        # evaluation (no validation)
        # ---------------------------
        elif args.finetune == False:
            trainer.predict(model, train_loader)
