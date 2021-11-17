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

from Dataset import Dataset
from Classifier import Classifier
from VGG_CNNC import VGG_CNNC
from SiameseVGG import SiameseVGG
from VGG import VGG

if __name__ == '__main__':

    def get_bool(x):
        if str(x).lower() == 'true':    return True
        elif str(x).lower() == 'false': return False
        else: raise Warning('Invalid boolean, using default')

    # ----------
    # arguments
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_seed', type=int)
    parser.add_argument('--datasets_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--load_datasets', type=get_bool, default=True)
    parser.add_argument('--data_type', type=str, default='scrna-seq')
    parser.add_argument('--do_training', type=get_bool, default=True)
    parser.add_argument('--do_testing', type=get_bool, default=False)
    parser.add_argument('--do_predict', type=get_bool, default=False)
    parser.add_argument('--do_finetune', type=get_bool, default=False)
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--train_split', type=float, default=.7)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--neighbors', type=int, default=2)
    parser.add_argument('--max_lag', type=int, default=5)
    parser.add_argument('--mask_lags', type=str, default='')
    parser.add_argument('--nbins_img', type=int, default=32)
    parser.add_argument('--mask_region', type=str, default='')
    parser.add_argument('--shuffle_traj', type=float, default=0.)
    parser.add_argument('--ncells_traj', type=int, default=0)
    parser.add_argument('--dropout_traj', type=float, default=0.)
    parser.add_argument('--auc_motif', type=str, default='none')
    parser.add_argument('--ablate_genes', type=get_bool, default=False)
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

    # ---------
    # set seed
    # ---------
    if args.load_datasets==False:
        args.global_seed = 1234
    pl.seed_everything(args.global_seed)

    # -------------------------
    # train_split (evaluation)
    # -------------------------
    if args.do_predict==True and args.do_finetune==False: args.train_split = 1.

    # ------------------
    # data type (fname)
    # ------------------
    if args.data_type=='scrna-seq': data_fname = 'ExpressionData.csv'
    elif args.data_type=='scatac-seq': data_fname = 'AccessibilityData.csv'

    # ------------------
    # data augmentation
    # ------------------
    if len(args.mask_lags)>0: args.mask_lags = [int(x) for x in args.mask_lags.split(',')]
    else: args.mask_lags = []

    # -------
    # prefix
    # -------
    if args.do_training==True: prefix = 'val_'
    elif args.do_testing==True: prefix = 'test_'
    elif args.do_predict==True: prefix = 'pred_'

    # ---------------------------
    # datasets (train/val split)
    # ---------------------------
    print('Loading datasets...')
    training, validation, val_names = [], [], []
    for item in tqdm(sorted(glob.glob(f'{args.datasets_dir}/*/*/*'))):
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
                           tf_ref=args.do_predict,
                           use_tf=(not args.do_finetune),
                           batchSize=args.batch_size,
                           load_prev=args.load_datasets)

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

    if args.load_datasets==False:
        sys.exit("Successfuly compiled datasets. To use, run with load_datasets==True.")

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
    if args.do_training==True:
        model = Classifier(args, backbone, val_names, prefix)
    else:
        model = Classifier.load_from_checkpoint(args.model_dir, hparams=args,
                backbone=backbone, val_names=val_names, prefix=prefix)

    # -------
    # logger
    # -------
    logger = TensorBoardLogger('lightning_logs', name=args.output_dir)

    # ----------
    # callbacks
    # ----------
    if args.do_training==True or args.do_finetune==True:
        if args.train_split < 1.:
            ckpt_fname = '{epoch}-{'+prefix+'avg_auprc:.3f}-{'+prefix+'avg_auroc:.3f}'
            monitor, mode = f'{prefix}avg_auc', 'max'
        else:
            monitor, mode, ckpt_fname = 'train_loss', 'min', '{epoch}-{train_loss:.6f}'

        callbacks = [ LearningRateMonitor(logging_interval='epoch'), ModelCheckpoint(
                      monitor=monitor, mode=mode, save_top_k=1,
                      dirpath=f"lightning_logs/{args.output_dir}/", filename=ckpt_fname) ]

    # -----------
    # pl trainer
    # -----------
    trainer = pl.Trainer(max_epochs=args.max_epochs, deterministic=True,
                         accelerator='ddp', gpus=args.num_gpus, #[1,],
                         logger=logger, callbacks=callbacks, num_sanity_val_steps=0,
                         plugins=[ DDPPlugin(find_unused_parameters=False) ],
                         check_val_every_n_epoch=args.check_val_every_n_epoch)

    # ------------------------
    # do_training (from init)
    # ------------------------
    if args.do_training==True:
        trainer.fit(model, train_loader, val_loader)

    # ------------------------------
    # do_testing (from pre-trained)
    # ------------------------------
    elif args.do_testing==True:
        trainer.test(model, val_loader)

        # ------------------------------
        # do_finetune (with validation)
        # ------------------------------
        if args.do_finetune==True:
            trainer.fit(model, train_loader, val_loader)

    # ---------------------------------
    # do_prediction (from pre-trained)
    # ---------------------------------
    elif args.do_predict==True:

        # -----------------------------
        # do_finetune (opt validation)
        # -----------------------------
        if args.do_finetune==True:
            if args.train_split==1.: trainer.fit(model, train_loader)
            else: trainer.fit(model, train_loader, val_loader)

        # ---------------------------
        # evaluation (no validation)
        # ---------------------------
        elif args.do_finetune==False:
            trainer.predict(model, train_loader)
