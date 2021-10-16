import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm import tqdm

import argparse
import os, sys
import glob
import pathlib
import shutil
import random
import math
import itertools
import pickle

from torch.utils.data                          import DataLoader
from torch.utils.data                          import random_split
from torch.utils.data                          import ConcatDataset
from pytorch_lightning.loggers                 import TensorBoardLogger
from pytorch_lightning.callbacks               import LearningRateMonitor
from pytorch_lightning.callbacks               import ModelCheckpoint
from pytorch_lightning.plugins                 import DDPPlugin
from pytorch_lightning.metrics.classification  import PrecisionRecallCurve as PRCurve
from pytorch_lightning.metrics.functional      import precision_recall_curve as prc
from pytorch_lightning.metrics.functional      import auc, roc

from vgg import VGG, VGG_CNNC, SiameseVGG

class Dataset(torch.utils.data.Dataset):
    """Generate/load batches of images"""
    def __init__(self,
                 root_dir,
                 rel_path,
                 neighbors,
                 max_lag,
                 mask_lags,
                 nbins,
                 mask_img='',
                 batchSize=None,
                 shuffle=0.,
                 ncells=0,
                 dropout=0.,
                 motif='none',
                 ablate=False,
                 tf_ref=False,
                 use_tf=False,
                 load_prev=True,
                 verbose=False):

        self.root_dir = root_dir
        self.rel_path = rel_path
        self.load_prev = load_prev
        self.batch_size = batchSize
        self.neighbors = neighbors
        self.max_lag = max_lag
        self.mask_lags = mask_lags
        self.nbins = nbins
        self.mask_img = mask_img
        self.shuffle = shuffle
        self.ncells = ncells
        self.dropout = dropout
        self.motif = motif
        self.ablate = ablate
        self.tf_ref = tf_ref
        self.use_tf = use_tf

        # get batch pathnames
        if self.load_prev==True:
            prev_path = '/'.join(self.rel_path.split('/')[:-1])+'*/'
            self.X_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'X_*.npy'))]
            self.y_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'y_*.npy'))]
            self.msk_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'msk_*.npy'))]
        else:
            # generate batches for each cell-type lineage (trajectory): X, y, mask
            self.sce_fnames = sorted(pathlib.Path(self.root_dir).glob(self.rel_path))
            self.X_fnames = [ None ] * len(self.sce_fnames)
            self.y_fnames = [ None ] * len(self.sce_fnames)
            self.msk_fnames = [ None ] * len(self.sce_fnames)
            for sce_fname in (tqdm(self.sce_fnames) if verbose else self.sce_fnames):
                self.generate_batches(str(sce_fname))

    def __len__(self):
        return len(self.X_fnames)

    def __getitem__(self, idx):
        X = np.load(self.X_fnames[idx], allow_pickle=True)
        y = np.load(self.y_fnames[idx], allow_pickle=True)
        msk = np.load(self.msk_fnames[idx], allow_pickle=True)
        return X, y, msk, self.X_fnames[idx].split('X_')

    def seed_from_string(self, s):
        """Generate random seed given a string"""
        n = int.from_bytes(s.encode(), 'little')
        return sum([int(x) for x in str(n)])

    def grouper(self, iterable, m, fillvalue=None):
        """Split iterable into chunks of size m"""
        args = [iter(iterable)] * m
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    def shuffle_pt(self, pt, seed=None, df=True):
        """Kernelized swapper"""
        if seed is not None: np.random.seed(seed)
        if df==True: pt = pt.copy().values
        for i in np.arange(pt.size):
            j = np.random.normal(loc=0, scale=self.shuffle*pt.size)
            i_ = int(round(np.clip(i+j, 0, pt.size-1)))
            pt[[i,i_]] = pt[[i_,i]]
        return pt

    def max_cross_correlation(self, a, v):
        corr = abs(np.correlate(a, v, "same"))
        return corr[ corr.size//2 - self.max_lag : corr.size//2 ].max()

    def generate_batches(self, sce_fname):
        """Generate batches as .npy files from sce file"""
        # generate batches for each lineage (trajectory)
        sce_folder = '/'.join(sce_fname.split('/')[:-1])
        pt = pd.read_csv(f"{sce_folder}/PseudoTime.csv", index_col=0)

        # loop over trajectories
        n_clusters = pt.shape[1]
        for k in range(n_clusters):
            traj_folder = f"{sce_folder}/traj{k+1}/"

            # remove previous results
            if os.path.isdir(traj_folder):
                shutil.rmtree(traj_folder)
            os.mkdir(traj_folder)

            # print message for dataset
            print(f"Generating batches for {'/'.join(sce_folder.split('/')[-2:])}...")

            # load single cell experiment from data file
            sce = pd.read_csv(sce_fname, index_col=0).T

            # convert gene names to lowercase
            sce.columns = sce.columns.str.lower()

            # get lineage (trajectory) indices, sorted by slingshot pseudotime values
            traj_idx = pt.iloc[np.where(~pt.iloc[:,k].isnull())[0], k].sort_values().index

            # shuffle pseudotime
            if self.shuffle > 0.:
                traj_idx = self.shuffle_pt(traj_idx, seed=None, df=True)

            # sample ncells (optional, only if traj > ncells)
            if self.ncells > 0 and self.ncells < traj_idx.size:
                traj_idx = traj_idx[np.sort(np.random.choice(np.arange(traj_idx.size), self.ncells, False))].copy()

            # additional dropouts
            if self.dropout > 0.:
                below_cutoff = (sce.loc[traj_idx,:].values < sce.loc[traj_idx,:].quantile(self.dropout, axis=1).values[:,None])
                drop = np.random.choice([0.,1.], p=[self.dropout,1-self.dropout], size=below_cutoff.shape)
                sce.loc[traj_idx,:] *= (below_cutoff.astype(int) * drop + (~below_cutoff).astype(int))

            # load gene regulation labels from reference file
            ref = pd.read_csv(f"{sce_folder}/refNetwork.csv")
            g1 = [g.lower() for g in ref['Gene1'].values]
            g2 = [g.lower() for g in ref['Gene2'].values]
            ref_1d = np.array(["%s %s" % x for x in list(zip(g1,g2))])

            # optionally, use TFs from separate reference file
            if self.tf_ref==True:
                tfs_ref = pd.read_csv(f"{sce_folder}/TranscriptionFactors.csv")
                tf = [g.lower() for g in tfs_ref['Gene1'].values]

                # optionally, use TFs as Gene1
                if self.use_tf==True: g1 = tf.copy()

            elif self.tf_ref==False:
                tf = g1.copy()

            # trajectory pairwise gene correlations (max absolute cross corr or max pearson corr)
            if self.max_lag > 0: traj_pcorr = sce.loc[traj_idx,:].corr(self.max_cross_correlation)
            elif self.max_lag==0: traj_pcorr = sce.loc[traj_idx,:].corr(method='pearson')

            # select gpairs in motif, optionally ablate (i.e. mask) genes
            gmasks, gpair_select = dict(), np.array([''])
            for g in itertools.product(sorted(set(g1)), sce.columns):
                if self.motif=='none':
                    gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                    gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
                elif self.motif=='ffl-reg':
                    pair_reg = ref.loc[ref['Gene2'].str.lower().isin(g), 'Gene1']
                    pair_reg = pair_reg[ pair_reg.duplicated() ].str.lower().values
                    if pair_reg.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                    if self.ablate==True: gmasks[g] = ~sce.columns.isin(pair_reg)
                    elif self.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
                elif self.motif=='ffl-tgt':
                    pair_tgt = ref.loc[ref['Gene1'].str.lower().isin(g), 'Gene2']
                    pair_tgt = pair_tgt[ pair_tgt.duplicated() ].str.lower().values
                    if pair_tgt.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                    if self.ablate==True: gmasks[g] = ~sce.columns.isin(pair_tgt)
                    elif self.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
                elif self.motif=='ffl-trans':
                    pair_trans = np.intersect1d(ref.loc[ref['Gene1'].str.lower()==g[0], 'Gene2'].str.lower().values,
                                                ref.loc[ref['Gene2'].str.lower()==g[1], 'Gene1'].str.lower().values)
                    if pair_trans.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                    if self.ablate==True: gmasks[g] = ~sce.columns.isin(pair_trans)
                    elif self.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
                elif self.motif=='fbl-trans':
                    pair_trans = np.intersect1d(ref.loc[ref['Gene1'].str.lower()==g[1], 'Gene2'].str.lower().values,
                                                    ref.loc[ref['Gene2'].str.lower()==g[0], 'Gene1'].str.lower().values)
                    if pair_trans.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                    if self.ablate==True: gmasks[g] = ~sce.columns.isin(pair_trans)
                    elif self.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
                elif self.motif=='mi-simple':
                    pair_si = ref.loc[(ref['Gene1'].str.lower()==g[1])&(ref['Gene2'].str.lower()==g[0]),:]
                    if pair_si.shape[0]>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                    gmasks[g] = np.ones((sce.shape[1],)).astype(bool)

            # generate list of tuples with all possible TF/target gpairs (opt, w/ neighbors)
            gpairs = [tuple( list(g) + list(traj_pcorr.loc[g[0],(traj_pcorr.index.isin(tf)) &
                                                                (~traj_pcorr.index.isin(g)) &
                                                                (gmasks[g])
                                                            ].nlargest(self.neighbors).index)
                                     + list(traj_pcorr.loc[g[1],(traj_pcorr.index.isin(tf)) &
                                                                (~traj_pcorr.index.isin(g)) &
                                                                (gmasks[g])
                                                            ].nlargest(self.neighbors).index) )
                      for g in itertools.product(sorted(set(g1)), sce.columns)]

            # shuffle order of TF/target gpairs
            seed = self.seed_from_string(traj_folder)
            random.seed(seed); random.shuffle(gpairs)

            # indices for pairwise comparisons
            gene_traj_pairs = [[0,1],[0,0],[1,1]]
            for i in range(self.neighbors):
                gene_traj_pairs.append([0,2+i])
                gene_traj_pairs.append([1,2+self.neighbors+i])

            # number of TF/target gpairs, no. cells
            n, n_cells = len(gpairs), traj_idx.size

            # generate TF/target gpair batches
            if self.batch_size is not None:
                gpairs_batched = [list(x) for x in self.grouper(gpairs, self.batch_size)]
                gpairs_batched = [list(filter(None, x)) for x in gpairs_batched]
            else: gpairs_batched = [gpairs]

            # loop over batches of TF/target gpairs
            for j in range(len(gpairs_batched)):
                X_fname = f"{traj_folder}/X_batch{j}_size{len(gpairs_batched[j])}.npy"
                y_fname = f"{traj_folder}/y_batch{j}_size{len(gpairs_batched[j])}.npy"
                msk_fname = f"{traj_folder}/msk_batch{j}_size{len(gpairs_batched[j])}.npy"
                g_fname = f"{traj_folder}/g_batch{j}_size{len(gpairs_batched[j])}.npy"

                # flatten TF/target gpairs (list of tuples) to list
                gpairs_list = list(itertools.chain(*gpairs_batched[j]))

                # split batch into single gpair examples (w/ neighbors)
                if self.batch_size is None or j==len(gpairs_batched)-1:
                    sce_list = np.array_split(sce.loc[traj_idx, gpairs_list].values, len(gpairs_batched[j]), axis=1)
                else:
                    sce_list = np.array_split(sce.loc[traj_idx, gpairs_list].values, self.batch_size, axis=1)

                # recombine re-shaped examples into full mini-batch
                sce_list = [g_sce.reshape(1,2+2*self.neighbors,1,n_cells) for g_sce in sce_list]
                X_batch = np.concatenate(sce_list, axis=0).astype(np.float32)

                # generate for batch: gene names, regulation labels, motif mask
                gpairs_batched_1d = np.array(["%s %s" % x[:2] for x in gpairs_batched[j]])
                y_batch = np.in1d(gpairs_batched_1d, ref_1d).reshape(X_batch.shape[0],1)
                msk_batch = np.in1d(gpairs_batched_1d, gpair_select).reshape(X_batch.shape[0],1)

                # generate 2D gene-gene co-expression images
                nchannels = len(gene_traj_pairs) * (1+self.max_lag)
                X_imgs = np.zeros((X_batch.shape[0], nchannels, self.nbins, self.nbins))

                # loop over examples in batch
                for i in range(X_imgs.shape[0]):

                    # loop over gene-gene pairwise comparisons
                    for pair_idx in range(len(gene_traj_pairs)):

                        # aligned gene-gene co-expression image
                        pair = gene_traj_pairs[pair_idx]
                        data = np.squeeze(X_batch[i,pair,:,:]).T
                        if 0 in self.mask_lags: pass
                        else:
                            H, _ = np.histogramdd(data, bins=(self.nbins,self.nbins))
                            H /= np.sqrt((H.flatten()**2).sum())
                            X_imgs[i,pair_idx*(1+self.max_lag),:,:] = H

                        # lagged gene-gene co-expression images
                        for lag in range(1,self.max_lag+1):
                            if lag in self.mask_lags: pass
                            else:
                                data_lagged = np.concatenate((data[:-lag,0].reshape(-1,1),
                                                              data[lag:,1].reshape(-1,1)), axis=1)
                                H, _ = np.histogramdd(data_lagged, bins=(self.nbins,self.nbins))
                                H /= np.sqrt((H.flatten()**2).sum())
                                X_imgs[i,pair_idx*(1+self.max_lag)+lag,:,:] = H

                # optionally, mask region(s)
                if self.mask_img=='off-off':
                    X_imgs[:,:,:(self.nbins//2),:(self.nbins//2)] = 0.
                if self.mask_img in ['on-off', 'on']:
                    X_imgs[:,:,(self.nbins//2):,:(self.nbins//2)] = 0.
                if self.mask_img in ['off-on', 'on']:
                    X_imgs[:,:,:(self.nbins//2),(self.nbins//2):] = 0.
                if self.mask_img in ['on-on', 'on']:
                    X_imgs[:,:,(self.nbins//2):,(self.nbins//2):] = 0.
                if self.mask_img=='edges':
                    X_imgs[:,:,0,:] = 0.; X_imgs[:,:,:,0] = 0.

                # save X, y, msk, g to pickled numpy files
                np.save(X_fname, X_imgs.astype(np.float32), allow_pickle=True)
                np.save(y_fname, y_batch.astype(np.float32), allow_pickle=True)
                np.save(msk_fname, msk_batch.astype(np.float32), allow_pickle=True)
                np.save(g_fname, gpairs_batched_1d.reshape(-1,1), allow_pickle=True)

                # save batch filenames for __len__ and __getitem__
                idx = np.where(np.array(self.X_fnames) == None)[0]
                if idx.size > 0:
                    self.X_fnames[idx[0]] = X_fname
                    self.y_fnames[idx[0]] = y_fname
                    self.msk_fnames[idx[0]] = msk_fname
                else:
                    self.X_fnames.extend([X_fname])
                    self.y_fnames.extend([y_fname])
                    self.msk_fnames.extend([msk_fname])

class Classifier(pl.LightningModule):
    """Deep neural network for binary classification of lagged gene-gene co-expression images"""
    def __init__(self, hparams, backbone, val_names, prefix):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.val_names = val_names
        self.prefix = prefix

        # store predictions & targets in pytortch_lightning precision-recall curve
        self.val_prc = nn.ModuleList([PRCurve(pos_label=1) for x in self.val_names])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr_init)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, train_batch, batch_idx):
        X, y, _, _ = train_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)
        loss = F.binary_cross_entropy_with_logits(
                    out, y, weight=y.sum()/y.size(0),
                    reduction='sum') / self.hparams.batch_size

        self.log('train_loss', loss,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True,
                 add_dataloader_idx=False)
        return loss

    def validation_step(self, val_batch, batch_idx, dataset_idx=0) -> None:
        X, y, _, _ = val_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)
        loss = F.binary_cross_entropy_with_logits(
                    out, y, weight=y.sum()/y.size(0),
                    reduction='sum') / self.hparams.batch_size

        # update precision-recall curve
        self.val_prc[dataset_idx](pred, y)

        self.log(f'{self.prefix}loss', loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 add_dataloader_idx=False)

    def test_step(self, test_batch, batch_idx, dataset_idx=0) -> None:
        X, y, msk, fname = test_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)

        pred_fname = fname[0] + f'pred_seed={self.hparams.global_seed}_' + fname[1]
        np.save(pred_fname, pred.cpu().detach().numpy().astype(np.float32), allow_pickle=True)

        if msk.sum()>0:
            # update precision-recall curve (opt mask)
            pred_msk = torch.masked_select(pred, msk>0)
            y_msk = torch.masked_select(y, msk>0)
            self.val_prc[dataset_idx](pred_msk, y_msk)

    def predict_step(self, pred_batch, batch_idx, dataset_idx=0) -> None:
        X, _, _, fname = pred_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)

        pred_fname = fname[0] + f'pred_seed={self.hparams.global_seed}_' + fname[1]
        np.save(pred_fname, pred.cpu().detach().numpy().astype(np.float32), allow_pickle=True)

    def on_validation_epoch_end(self):
        val_auprc = torch.zeros((len(self.val_names),),
                    device=torch.cuda.current_device())
        val_auroc = torch.zeros((len(self.val_names),),
                    device=torch.cuda.current_device())
        for idx in range(len(self.val_names)):
            name = self.val_names[idx]
            # NOTE: currently, AUC averaged across processes
            preds = torch.cat(self.val_prc[idx].preds, dim=0)
            target = torch.cat(self.val_prc[idx].target, dim=0)
            precision, recall, _ = prc(preds, target, pos_label=1)
            fpr, tpr, _ = roc(preds, target, pos_label=1)
            auprc, auroc = auc(recall, precision), auc(fpr, tpr)
            val_auprc[idx] = auprc; val_auroc[idx] = auroc
            self.log(f'{name}_auprc', auprc,
                     sync_dist=True,
                     add_dataloader_idx=False)
            self.log(f'{name}_auroc', auroc,
                     sync_dist=True,
                     add_dataloader_idx=False)
            self.val_prc[idx].reset()

        avg_auroc = val_auroc.mean()
        avg_auprc = val_auprc.mean()
        avg_auc = (avg_auroc + avg_auprc)/2
        self.log(f'{self.prefix}avg_auroc', avg_auroc,
                 sync_dist=True,
                 add_dataloader_idx=False)
        self.log(f'{self.prefix}avg_auprc', avg_auprc,
                 sync_dist=True,
                 add_dataloader_idx=False)
        self.log(f'{self.prefix}avg_auc', avg_auc,
                 sync_dist=True,
                 add_dataloader_idx=False)

    def on_test_epoch_end(self):
        test_auprc = torch.zeros((len(self.val_names),),
                     device=torch.cuda.current_device())
        test_auroc = torch.zeros((len(self.val_names),),
                     device=torch.cuda.current_device())
        for idx in range(len(self.val_names)):
            name = self.val_names[idx]
            # NOTE: currently, AUC averaged across processes
            preds = torch.cat(self.val_prc[idx].preds, dim=0)
            target = torch.cat(self.val_prc[idx].target, dim=0)
            precision, recall, _ = prc(preds, target, pos_label=1)
            fpr, tpr, _ = roc(preds, target, pos_label=1)
            auprc, auroc = auc(recall, precision), auc(fpr, tpr)
            test_auprc[idx] = auprc; test_auroc[idx] = auroc
            density = target.sum() / target.size(0)
            self.log(f'_{name}_auprc', auprc,
                     sync_dist=True,
                     add_dataloader_idx=False)
            self.log(f'_{name}_auroc', auroc,
                     sync_dist=True,
                     add_dataloader_idx=False)
            self.log(f'_{name}_density', density,
                     sync_dist=True,
                     add_dataloader_idx=False)
            self.val_prc[idx].reset()

        avg_auroc = test_auroc.mean()
        avg_auprc = test_auprc.mean()
        avg_auc = (avg_auroc + avg_auprc)/2
        self.log(f'_{self.prefix}avg_auroc', avg_auroc,
                 sync_dist=True,
                 add_dataloader_idx=False)
        self.log(f'_{self.prefix}avg_auprc', avg_auprc,
                 sync_dist=True,
                 add_dataloader_idx=False)
        self.log(f'_{self.prefix}avg_auc', avg_auc,
                 sync_dist=True,
                 add_dataloader_idx=False)

# ------------
# main script
# ------------
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
    parser.add_argument('--output_dir', type=str)
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
    # train split (prediction)
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

    # -------
    # prefix
    # -------
    if args.do_training==True: prefix = 'val_'
    elif args.do_testing==True: prefix = 'test_'
    elif args.do_predict==True: prefix = 'pred_'

    # ----------------------------
    # model (init or pre-trained)
    # ----------------------------
    if args.do_training==True:
        model = Classifier(args, backbone, val_names, prefix)
    else:
        model = Classifier.load_from_checkpoint(args.model_dir,
                backbone=backbone, val_names=val_names, prefix=prefix)

    # -------
    # logger
    # -------
    logger = TensorBoardLogger('lightning_logs', name=args.output_dir)

    # ----------
    # callbacks
    # ----------
    if args.do_training==True or args.do_finetune==True:
        ckpt_fname = '{epoch}-{' + prefix + 'avg_auprc:.3f}-{' + prefix + 'avg_auroc:.3f}'
        callbacks = [ LearningRateMonitor(logging_interval='epoch'),
                      ModelCheckpoint(monitor=f'{prefix}avg_auc', mode='max', save_top_k=1,
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

        # ------------
        # do_finetune
        # ------------
        if args.do_finetune==True:
            trainer.fit(model, train_loader, val_loader)

    # ---------------------------------
    # do_prediction (from pre-trained)
    # ---------------------------------
    elif args.do_predict==True:

        # ------------
        # do_finetune
        # ------------
        if args.do_finetune==True:
            trainer.fit(model, train_loader, val_loader)

        elif args.do_finetune==False:
            trainer.predict(model, train_loader)
