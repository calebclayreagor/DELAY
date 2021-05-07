import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn.functional as F

import argparse
import os, sys
import glob, pathlib
import shutil
import random, math
import itertools

from torch.utils.data                          import DataLoader
from pytorch_lightning.loggers                 import TensorBoardLogger
from pytorch_lightning.callbacks               import LearningRateMonitor
from pytorch_lightning.plugins                 import DDPPlugin
from pytorch_lightning.metrics.classification  import PrecisionRecallCurve as PRCurve
from pytorch_lightning.metrics.functional      import auc, precision_recall_curve as prc, roc


class Dataset(torch.utils.data.Dataset):
    """Dataset class for generating/loading batches"""
    def __init__(self, root_dir, rel_path, context_dims, batchSize=None, wShuffle=0.02,
                 minCells=40, overwrite=False, load_prev=True, verbose=False):

        self.root_dir = root_dir
        self.rel_path = rel_path
        self.context_dims = context_dims
        self.batch_size = batchSize
        self.overwrite = overwrite
        self.load_prev = load_prev
        self.pt_shuffle = wShuffle
        self.min_cells = minCells

        if self.load_prev==True:
            # ----------------------------------
            # get batch pathnames (X, y, y_aux)
            # ----------------------------------
            prev_path = '/'.join(self.rel_path.split('/')[:-1])+'*/'
            self.X_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'X_*.npy'))]
            self.y_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'y_b*.npy'))]
            self.y_aux_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'y_a*.npy'))]

        else:
            # -----------------------------------------------------
            # generate batch(es) for each trajectory (X, y, y_aux)
            # -----------------------------------------------------
            self.sce_fnames = sorted(pathlib.Path(self.root_dir).glob(self.rel_path))
            self.X_fnames = [None] * len(self.sce_fnames)
            self.y_fnames = [None] * len(self.sce_fnames)
            self.y_aux_fnames = [None] * len(self.sce_fnames)
            for sce_fname in (tqdm(self.sce_fnames) if verbose else self.sce_fnames):
                self.generate_batches(str(sce_fname))


    def __len__(self):
        """Total number of batches"""
        return len(self.X_fnames)


    def __getitem__(self, idx):
        """Load a given batch if trajectory size > min_cells"""
        new_batch, idx_ = True, idx
        np.random.seed(self.seed_from_string(self.X_fnames[idx]))
        while new_batch:
            fname = self.X_fnames[idx_]
            X = np.load(self.X_fnames[idx_], allow_pickle=True)
            y = np.load(self.y_fnames[idx_], allow_pickle=True)
            y_aux = np.load(self.y_aux_fnames[idx_], allow_pickle=True)
            if X.shape[3] > self.min_cells: new_batch = False
            idx_ = np.random.choice(np.arange(len(self.X_fnames)))
        return X, y, y_aux, fname


    def seed_from_string(self, s):
        """Generate random seed given a string"""
        n = int.from_bytes(s.encode(), 'little')
        return sum([int(x) for x in str(n)])


    def grouper(self, iterable, m, fillvalue=None):
        """Split iterable into chunks of size m"""
        args = [iter(iterable)] * m
        return itertools.zip_longest(*args, fillvalue=fillvalue)


    def shuffle_pt(self, pt, seed):
        """Kernelized swapper"""
        np.random.seed(seed)
        pt = pt.copy().values
        for i in np.arange(pt.size):
            j = np.random.normal(loc=0, scale=self.pt_shuffle*pt.size)
            i_ = int(round(np.clip(i+j, 0, pt.size-1)))
            pt[[i,i_]] = pt[[i_,i]]
        return pt

    def max_cross_correlation(self, a, v, mode='same'):
        return abs(np.correlate(a, v, mode)).max()


    def generate_batches(self, sce_fname):
        """Generate batch(es) as .npy file(s) from sce"""
        # generate batch(es) (>=1) for each trajectory
        sce_folder = '/'.join(sce_fname.split('/')[:-1])
        pt = pd.read_csv(f"{sce_folder}/PseudoTime.csv", index_col=0)
        n_clusters, sce, ref, g1, g2 = pt.shape[1], None, None, None, None

        # outer loop over trajectories
        for k in range(n_clusters):
            traj_folder = f"{sce_folder}/traj{k+1}/"

            # load previously generated batches for given trajectory
            if os.path.isdir(traj_folder) and self.overwrite==False:
                # save batch filenames for __len__ and __getitem__
                for file in sorted(glob.glob(f'{traj_folder}/*.npy')):

                    # save batch filename for expression data X
                    if file.split('/')[-1][0]=='X':
                        idx = np.where(np.array(self.X_fnames) == None)[0]
                        if idx.size > 0:
                            self.X_fnames[idx[0]] = file
                        else:
                            self.X_fnames.extend([file])

                    # save batch filename for y_aux = bool(is TF?)
                    elif file.split('/')[-1][:5]=='y_aux':
                        idx = np.where(np.array(self.y_aux_fnames) == None)[0]
                        if idx.size > 0:
                            self.y_aux_fnames[idx[0]] = file
                        else:
                            self.y_aux_fnames.extend([file])

                    # save batch filename for regulation labels y
                    elif file.split('/')[-1][0]=='y':
                        idx = np.where(np.array(self.y_fnames) == None)[0]
                        if idx.size > 0:
                            self.y_fnames[idx[0]] = file
                        else:
                            self.y_fnames.extend([file])
            else:
                # remove previous results
                if os.path.isdir(traj_folder):
                    shutil.rmtree(traj_folder)
                os.mkdir(traj_folder)

                if sce is not None: pass
                else:
                    # load single cell experiment data from file
                    sce = pd.read_csv(sce_fname, index_col=0).T

                    # sort expression in experiment using slingshot pseudotime
                    sce = sce.loc[pt.sum(axis=1).sort_values().index,:].copy()

                    # if synthetic experiment, shuffle pseudotime
                    if sce_folder.split('/')[-4]!='experimental':
                        seed = self.seed_from_string(traj_folder)
                        sce = sce.loc[self.shuffle_pt(sce.index, seed),:].copy()

                # trajectory pairwise gene correlations: max absolute cross correlation
                traj_idx = np.where(~pt.iloc[:,k].isnull())[0]
                if traj_idx.size >= self.min_cells:
                    traj_pcorr = sce.iloc[traj_idx,:].corr(self.max_cross_correlation)
                else: traj_pcorr = sce.corr(self.max_cross_correlation)

                # generate list of tuples containing all possible gene pairs + context
                gpairs = [tuple(list(g)+list(traj_pcorr.loc[g[1],~traj_pcorr.index.isin(g)].nlargest(self.context_dims).index)) for g in itertools.product(sce.columns, repeat=2)]
                seed = self.seed_from_string(traj_folder); random.seed(seed); random.shuffle(gpairs)

                n, n_cells = len(gpairs), sce.shape[0]

                if ref is not None: pass
                else:
                    # load gene regulation labels from reference file
                    ref = pd.read_csv(f"{sce_folder}/refNetwork.csv")
                    g1, g2 = ref['Gene1'].values, ref['Gene2'].values
                    ref_1d = np.array(["%s %s" % x for x in list(zip(g1,g2))])

                # split gene pairs + context into batches
                if self.batch_size is not None:
                    gpairs_batched = [list(x) for x in self.grouper(gpairs, self.batch_size)]
                    gpairs_batched = [list(filter(None, x)) for x in gpairs_batched]
                else: gpairs_batched = [gpairs]

                # print message if generating batches for experimental dataset
                if sce_folder.split('/')[-4]=='experimental':
                    print(f"Generating batches for {'/'.join(sce_folder.split('/')[-2:])}")

                # inner loop over batches of gene pairs
                for j in range(len(gpairs_batched)):
                    X_fname = f"{traj_folder}/X_batch{j}_size{len(gpairs_batched[j])}.npy"
                    y_fname = f"{traj_folder}/y_batch{j}_size{len(gpairs_batched[j])}.npy"
                    y_aux_fname = f"{traj_folder}/y_aux_batch{j}_size{len(gpairs_batched[j])}.npy"

                    # flatten gene pairs + context (list of tuples) to list
                    gpairs_list = list(itertools.chain(*gpairs_batched[j]))

                    # split minibatch into single examples of gene A -> geneB + context
                    if self.batch_size is None or j==len(gpairs_batched)-1:
                        sce_list = np.array_split(sce[gpairs_list].values, len(gpairs_batched[j]), axis=1)
                    else:
                        sce_list = np.array_split(sce[gpairs_list].values, self.batch_size, axis=1)

                    # recombine re-shaped examples into full mini-batch
                    sce_list = [g_sce.reshape(1,1,2+self.context_dims,n_cells) for g_sce in sce_list]
                    X_batch = np.concatenate(sce_list, axis=0).astype(np.float32)

                    # generate gene regulation labels y for mini-batch
                    gpairs_batched_1d = np.array(["%s %s" % x[:2] for x in gpairs_batched[j]])
                    y_batch = np.in1d(gpairs_batched_1d, ref_1d).reshape(X_batch.shape[0],1)

                    # generate gene A transcription factor labels y_aux for mini-batch
                    geneA_batched_1d = np.array([x[0] for x in gpairs_batched[j]])
                    y_aux_batch = np.in1d(geneA_batched_1d, g1).reshape(X_batch.shape[0],1)

                    # select and sort cells in current trajectory
                    X_normalized = X_batch[...,traj_idx]
                    y_float = y_batch.astype(np.float32)
                    y_aux_float = y_aux_batch.astype(np.float32)
                    if X_normalized.shape[-1] > 0:
                        X_normalized /= np.quantile(X_normalized, 0.95)

                    # save X, y, y_aux to pickled numpy files
                    np.save(X_fname, X_normalized, allow_pickle=True)
                    np.save(y_fname, y_float, allow_pickle=True)
                    np.save(y_aux_fname, y_aux_float, allow_pickle=True)

                    # save batch filenames for __len__ and __getitem__
                    idx = np.where(np.array(self.X_fnames) == None)[0]
                    if idx.size > 0:
                        self.X_fnames[idx[0]] = X_fname
                        self.y_fnames[idx[0]] = y_fname
                        self.y_aux_fnames[idx[0]] = y_aux_fname
                    else:
                        self.X_fnames.extend([X_fname])
                        self.y_fnames.extend([y_fname])
                        self.y_aux_fnames.extend([y_aux_fname])


class ConvNet(torch.nn.Module):
    def __init__(self, hidden_dim, context_dims, pyramid_dims, dropout):
        super(ConvNet, self).__init__()

        Activation = torch.nn.ReLU
        self.pyrd = pyramid_dims

        # ----------------
        # dense block 1.1
        # ----------------
        self.block11_branch1_1x3_3x1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, hidden_dim, padding=(0,1), kernel_size=(1,3)), Activation(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        self.block11_branch2_1x7_3x1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, hidden_dim, padding=(0,3), kernel_size=(1,7)), Activation(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        self.block11_branch3_maxpool = torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.block11_branch4_1x1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, kernel_size=(1,1)), Activation())

        # ---------------------
        # block 1.1 transition
        # ---------------------
        self.block11_transition_1x1 = torch.nn.Sequential(
        torch.nn.BatchNorm2d(2*hidden_dim+2),
        torch.nn.Dropout2d(p=dropout),
        torch.nn.Conv2d(2*hidden_dim+2, hidden_dim, kernel_size=(1,1)), Activation())

        # ----------------
        # dense block 1.2
        # ----------------
        self.block12_branch1_1x3_3x1 = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(0,1), kernel_size=(1,3)), Activation(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        self.block12_branch2_1x7_3x1 = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(0,3), kernel_size=(1,7)), Activation(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        self.block12_branch3_maxpool = torch.nn.Sequential(
        torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        torch.nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)), Activation())
        self.block12_branch4_1x1 = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)), Activation())

        # ---------------------
        # block 1.2 transition
        # ---------------------
        self.block12_transition_1x1 = torch.nn.Sequential(
        torch.nn.BatchNorm2d(2*hidden_dim+2),
        torch.nn.Dropout2d(p=dropout),
        torch.nn.Conv2d(2*hidden_dim+2, hidden_dim, kernel_size=(1,1)), Activation())

        # ----------------
        # dense block 1.3
        # ----------------
        self.block13_branch1_1x3_3x1 = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(0,1), kernel_size=(1,3)), Activation(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        self.block13_branch2_1x7_3x1 = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(0,3), kernel_size=(1,7)), Activation(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        self.block13_branch3_maxpool = torch.nn.Sequential(
        torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        torch.nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)), Activation())
        self.block13_branch4_1x1 = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)), Activation())

        # # -----------------------
        # # dense block transition
        # # -----------------------
        # self.dense_transition_layer_1x1_reduce = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2)*3, 1,  kernel_size=(1,1)), Activation(),
        # torch.nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)))

        # # ----------------
        # # dense block 2.1
        # # ----------------
        # self.block21_branch1_1x3_3x1 = torch.nn.Sequential(
        # torch.nn.Conv2d(1, hidden_dim, padding=(0,1), kernel_size=(1,3)), Activation(),
        # torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        # self.block21_branch2_1x7_3x1 = torch.nn.Sequential(
        # torch.nn.Conv2d(1, hidden_dim, padding=(0,3), kernel_size=(1,7)), Activation(),
        # torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        # self.block21_branch3_maxpool = torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.block21_branch4_1x1_self = torch.nn.Sequential(
        # torch.nn.Conv2d(1, 1, kernel_size=(1,1)), Activation())

        # # ----------------
        # # dense block 2.2
        # # ----------------
        # self.block22_branch1_1x1_1x3_3x1 = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2), 1, kernel_size=(1,1)), Activation(),
        # torch.nn.Conv2d(1, hidden_dim, padding=(0,1), kernel_size=(1,3)), Activation(),
        # torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        # self.block22_branch2_1x1_1x7_3x1 = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2), 1, kernel_size=(1,1)), Activation(),
        # torch.nn.Conv2d(1, hidden_dim, padding=(0,3), kernel_size=(1,7)), Activation(),
        # torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        # self.block22_branch3_maxpool_1x1 = torch.nn.Sequential(
        # torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        # torch.nn.Conv2d((2*hidden_dim+2), 1, kernel_size=(1,1)), Activation())
        # self.block22_branch4_1x1_reduce = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2), 1, kernel_size=(1,1)), Activation())

        # # ----------------
        # # dense block 2.3
        # # ----------------
        # self.block23_branch1_1x1_1x3_3x1 = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2)*2, 1, kernel_size=(1,1)), Activation(),
        # torch.nn.Conv2d(1, hidden_dim, padding=(0,1), kernel_size=(1,3)), Activation(),
        # torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        # self.block23_branch2_1x1_1x7_3x1 = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2)*2, 1, kernel_size=(1,1)), Activation(),
        # torch.nn.Conv2d(1, hidden_dim, padding=(0,3), kernel_size=(1,7)), Activation(),
        # torch.nn.Conv2d(hidden_dim, hidden_dim, padding=(1,0), kernel_size=(3,1)), Activation())
        # self.block23_branch3_maxpool_1x1 = torch.nn.Sequential(
        # torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        # torch.nn.Conv2d((2*hidden_dim+2)*2, 1, kernel_size=(1,1)), Activation())
        # self.block23_branch4_1x1_reduce = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2)*2, 1, kernel_size=(1,1)), Activation())

        # ------------------------
        # final transition layers
        # ------------------------
        self.dense1_final_transition_2x1_reduce = torch.nn.Sequential(
        torch.nn.BatchNorm2d((2*hidden_dim+2)*1),
        torch.nn.Dropout2d(p=dropout),
        torch.nn.Conv2d((2*hidden_dim+2)*1, hidden_dim, kernel_size=(2+context_dims,1)), Activation())
        # self.dense2_final_transition_2x1_reduce = torch.nn.Sequential(
        # torch.nn.Conv2d((2*hidden_dim+2)*3, (2*hidden_dim+2)*3, kernel_size=(2+context_dims,1)), Activation())

        # self.transformer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim, dropout=0)

        # ----------------------
        # fully connected layer
        # ----------------------
        self.fc_output = torch.nn.Sequential(
        # torch.nn.Dropout(p=dropout),
        # torch.nn.Linear(((2*hidden_dim+2)*1)*1, hidden_dim), Activation(),
        torch.nn.Linear(hidden_dim, 1))

    # ---------
    # forward
    # ---------
    def forward(self, x):
        # ----------
        # block 1.1
        # ----------
        out111 = self.block11_branch1_1x3_3x1(x)
        out112 = self.block11_branch2_1x7_3x1(x)
        out113 = self.block11_branch3_maxpool(x)
        out114 = self.block11_branch4_1x1(x)
        out11 = torch.cat([out111,out112,out113,out114], axis=1)

        # ---------------------
        # block 1.1 transition
        # ---------------------
        out = self.block11_transition_1x1(out11)

        # ----------
        # block 1.2
        # ----------
        out121 = self.block12_branch1_1x3_3x1(out)
        out122 = self.block12_branch2_1x7_3x1(out)
        out123 = self.block12_branch3_maxpool(out)
        out124 = self.block12_branch4_1x1(out)
        out12 = torch.cat([out121,out122,out123,out124], axis=1)
        # out1 = torch.cat([out11, out12], axis=1)

        # ---------------------
        # block 1.2 transition
        # ---------------------
        out = self.block12_transition_1x1(out12)

        # ----------
        # block 1.3
        # ----------
        out131 = self.block13_branch1_1x3_3x1(out)
        out132 = self.block13_branch2_1x7_3x1(out)
        out133 = self.block13_branch3_maxpool(out)
        out134 = self.block13_branch4_1x1(out)
        out13 = torch.cat([out131,out132,out133,out134], axis=1)
        # out1 = torch.cat([out11,out12,out13], axis=1)

        # # -----------------------
        # # dense block transition
        # # -----------------------
        # out = self.dense_transition_layer_1x1_reduce(out1)

        # # ----------
        # # block 2.1
        # # ----------
        # out211 = self.block21_branch1_1x3_3x1(out)
        # out212 = self.block21_branch2_1x7_3x1(out)
        # out213 = self.block21_branch3_maxpool(out)
        # out214 = self.block21_branch4_1x1_self(out)
        # out21 = torch.cat([out211,out212,out213,out214], axis=1)

        # # ----------
        # # block 2.2
        # # ----------
        # out221 = self.block22_branch1_1x1_1x3_3x1(out21)
        # out222 = self.block22_branch2_1x1_1x7_3x1(out21)
        # out223 = self.block22_branch3_maxpool_1x1(out21)
        # out224 = self.block22_branch4_1x1_reduce(out21)
        # out22 = torch.cat([out221,out222,out223,out224], axis=1)
        # out2 = torch.cat([out21, out22], axis=1)

        # # ----------
        # # block 2.3
        # # ----------
        # out231 = self.block23_branch1_1x1_1x3_3x1(out2)
        # out232 = self.block23_branch2_1x1_1x7_3x1(out2)
        # out233 = self.block23_branch3_maxpool_1x1(out2)
        # out234 = self.block23_branch4_1x1_reduce(out2)
        # out23 = torch.cat([out231,out232,out233,out234], axis=1)
        # out2 = torch.cat([out21,out22,out23], axis=1)

        # -----------------
        # final transition
        # -----------------
        out1 = self.dense1_final_transition_2x1_reduce(out13)
        # out2 = self.dense2_final_transition_2x1_reduce(out2)

        # ------------------
        # pyramidal pooling
        # ------------------
        out1 = F.avg_pool2d(out1, kernel_size=(1,out1.size()[-1]//self.pyrd))
        # out2 = F.avg_pool2d(out2, kernel_size=(1,out2.size()[-1]//self.pyrd))
        # out = torch.cat([out1[...,:self.pyrd],out2[...,:self.pyrd]], axis=1)
        out = torch.squeeze(F.max_pool2d(out1, kernel_size=(1,out1.size()[-1])))

        # out = torch.unsqueeze(out, 0)
        # out = torch.squeeze(self.transformer(out))

        # ----------------------
        # fully connected layer
        # ----------------------
        return self.fc_output(out)


# class GatedNet(torch.nn.Module):
#     def __init__(self, hidden_dim, context_dims, dropout):
#         super(GatedNet, self).__init__()
#
#         self.gated_unit = torch.nn.LSTM(2+context_dims,
#                                         hidden_dim,
#                                         num_layers=2,
#                                         dropout=dropout,
#                                         batch_first=True)
#         self.prediction = torch.nn.Linear(hidden_dim, 1)
#
#     # ---------
#     # forward
#     # ---------
#     def forward(self, x):
#         out = torch.squeeze(x)
#         out = torch.swapaxes(out, 1, 2)
#         out = self.gated_unit(out)
#         out = out[1][1][-1,...]
#         return self.prediction(out)


class Classifier(pl.LightningModule):
    """Convolutional network for binary classification of gene trajectory pairs"""
    def __init__(self, hparams, backbone, val_names, test_names):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.backbone = backbone

        # -------------------------------------------------
        # store preds & targets in precision-recall curves
        # -------------------------------------------------
        self.val_names, self.test_names = val_names, test_names
        self.val_prc = torch.nn.ModuleList([PRCurve(pos_label=1) for x in self.val_names])
        self.test_prc = torch.nn.ModuleList([PRCurve(pos_label=1) for x in self.test_names])

    # -------------------------
    # optimizer & LR scheduler
    # -------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr_init, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.hparams.lr_step)
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler, 'monitor' : 'val_loss_batch_exp'}

    # --------------------
    # weighted focal loss
    # --------------------
    def focal_loss(self, output, labels):
        """Compute weighted focal loss with given alpha, gamma hyperparameters"""
        alpha = min([(1 - labels).sum() / labels.size()[0], self.hparams.max_alpha])
        at = (labels * alpha) + ((1 - labels) * (1 - alpha))
        logpt = -F.binary_cross_entropy_with_logits(output, labels, reduction='none')
        return -at * (1 - logpt.exp()) ** self.hparams.gamma * logpt, alpha

    # -------------
    # forward pass
    # -------------
    def forward(self, x):
        return self.backbone(x)

    # --------------
    # training step
    # --------------
    def training_step(self, train_batch, batch_idx):
        """Optimize summed losses over batch"""
        X, y, y_aux, fname = train_batch
        out = self.forward(X)
        loss, alpha = self.focal_loss(out, y)
        loss_sum = loss.sum()

        # ------------
        # update logs
        # ------------
        if fname.split('/')[-6]=='experimental':
            self.log('train_loss_batch_exp',
                     loss_sum,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True)
        else:
            self.log('train_loss_batch_synt',
                     loss_sum,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True)

        return loss_sum

    # ----------------
    # validation step
    # ----------------
    def validation_step(self, val_batch, batch_idx, dataset_idx) -> None:
        """Val step updates metrics for each model across all datasets"""
        X, y, y_aux, fname = val_batch
        out = self.forward(X)
        loss, _ = self.focal_loss(out, y)
        loss_sum = loss.sum()
        pred = torch.sigmoid(out)

        # ----------------
        # update PR curve
        # ----------------
        if torch.sum(y_aux) > 0:
            y_aux_bool = y_aux.type(torch.bool)
            pred_tf = torch.masked_select(pred, y_aux_bool)
            y_tf = torch.masked_select(y, y_aux_bool)
            self.val_prc[dataset_idx](pred_tf, y_tf)

        # ------------
        # update logs
        # ------------
        if fname.split('/')[-6]=='experimental':
            self.log('val_loss_batch_exp',
                     loss_sum,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True,
                     add_dataloader_idx=False)
        else:
            self.log('val_loss_batch_synt',
                     loss_sum,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True,
                     add_dataloader_idx=False)


    # ---------------------
    # validation epoch end
    # ---------------------
    def on_validation_epoch_end(self):
        """Compute AUPRC & AUROC for each model/experiment"""
        for idx in range(len(self.val_names)):
            name = self.val_names[idx]
            # NOTE: currently, AUC averaged across processes
            preds = torch.cat(self.val_prc[idx].preds, dim=0)
            target = torch.cat(self.val_prc[idx].target, dim=0)
            precision, recall, _ = prc(preds, target, pos_label=1)
            self.log(f'{name}_auprc', auc(recall, precision),
                     sync_dist=True, add_dataloader_idx=False)
            fpr, tpr, _ = roc(preds, target, pos_label=1)
            self.log(f'{name}_auroc', auc(fpr, tpr),
                     sync_dist=True, add_dataloader_idx=False)
            self.val_prc[idx].reset()

    # ----------
    # test step
    # ----------
    def test_step(self, test_batch, batch_idx, dataset_idx) -> None:
        """Test step used to update metrics on individual datasets"""
        X, y, y_aux, fname = test_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)

        # ----------------
        # update PR curve
        # ----------------
        if torch.sum(y_aux) > 0:
            y_aux_bool = y_aux.type(torch.bool)
            pred_tf = torch.masked_select(pred, y_aux_bool)
            y_tf = torch.masked_select(y, y_aux_bool)
            self.test_prc[dataset_idx](pred_tf, y_tf)

    # ---------------
    # test epoch end
    # ---------------
    def on_test_epoch_end(self):
        """Compute AUPRC & AUROC for each dataset"""
        for idx in range(len(self.test_names)):
            name = self.test_names[idx]
            # NOTE: currently, AUC averaged across processes
            preds = torch.cat(self.test_prc[idx].preds, dim=0)
            target = torch.cat(self.test_prc[idx].target, dim=0)
            precision, recall, _ = prc(preds, target, pos_label=1)
            self.log(f'{name}_auprc', auc(recall, precision),
                     sync_dist=True, add_dataloader_idx=False)
            fpr, tpr, _ = roc(preds, target, pos_label=1)
            self.log(f'{name}_auroc', auc(fpr, tpr),
                     sync_dist=True, add_dataloader_idx=False)
            self.test_prc[idx].reset()


# ----------
# main script
# ----------
if __name__ == '__main__':
    # -----
    # seed
    # -----
    pl.seed_everything(1234)

    # ----------
    # arguments
    # ----------
    def get_bool(x):
        if str(x).lower() == 'true':    return True
        elif str(x).lower() == 'false': return False
        else: raise Warning('Invalid boolean, using default')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--finetune', type=get_bool, default=False)
    parser.add_argument('--ovr_train', type=get_bool, default=False)
    parser.add_argument('--ovr_val', type=get_bool, default=False)
    parser.add_argument('--ovr_test', type=get_bool, default=False)
    parser.add_argument('--ovr_tune', type=get_bool, default=False)
    parser.add_argument('--load_prev', type=get_bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--context_dims', type=int, default=0)
    parser.add_argument('--lr_init', type=float, default=0.00001)
    parser.add_argument('--lr_step', type=int, default=30)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--tune_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--max_alpha', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=12)
    parser.add_argument('--pyramid_dims', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad_clip_val', type=float, default=1.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--num_workers', type=int, default=36)
    parser.add_argument('--num_gpus', type=int, default=2)
    args = parser.parse_args()

    # -----------------
    # training dataset
    # -----------------
    print('Loading training batches...')
    training = Dataset(root_dir=f'{args.dataset_dir}/training', rel_path='*/*/*/*/ExpressionData.csv',
                       context_dims=args.context_dims, batchSize=args.batch_size, overwrite=args.ovr_train,
                                                                    load_prev=args.load_prev, verbose=True)
    train_loader = DataLoader(training, batch_size=None, shuffle=True, num_workers=args.num_workers)

    # -------------
    # tune dataset
    # -------------
    if args.finetune==True:
        print('Loading fine-tune batches...')
        training_tune = Dataset(root_dir=f'{args.dataset_dir}/training_tune', rel_path='*/*/*/*/ExpressionData.csv',
                                context_dims=args.context_dims, batchSize=args.batch_size, overwrite=args.ovr_tune,
                                                                             load_prev=args.load_prev, verbose=True)
        tune_loader = DataLoader(training_tune, batch_size=None, shuffle=True, num_workers=args.num_workers)

    # --------------------
    # validation datasets
    # --------------------
    validation, val_names = [], []
    print('Loading validation datasets...')
    for item in tqdm(sorted(glob.glob(f'{args.dataset_dir}/validation/*/*/*'))):
        if os.path.isdir(item):
            val_names.append('val_'+'_'.join(item.split('/')[-2:]))
            validation.append(Dataset(root_dir=item, rel_path='*/ExpressionData.csv',
            context_dims=args.context_dims, batchSize=args.batch_size,
            overwrite=args.ovr_val, load_prev=args.load_prev))

    val_loader = [None] * len(validation)
    for i in range(len(validation)):
        val_loader[i] = DataLoader(validation[i], batch_size=None, num_workers=args.num_workers)

    # -----------------
    # testing datasets
    # -----------------
    testing, test_names = [], []
    print('Loading testing datasets...')
    for item in tqdm(sorted(glob.glob(f'{args.dataset_dir}/testing/*/*/*/*'))):
        if os.path.isdir(item):
            test_names.append('test_'+'_'.join(item.split('/')[-2:]))
            testing.append(Dataset(root_dir=item, rel_path='ExpressionData.csv',
            context_dims=args.context_dims, batchSize=args.batch_size,
            overwrite=args.ovr_val, load_prev=args.load_prev))

    test_loader = [None] * len(testing)
    for i in range(len(testing)):
        test_loader[i] = DataLoader(testing[i], batch_size=None, num_workers=args.num_workers)

    # -------
    # logger
    # -------
    logger = TensorBoardLogger('lightning_logs', name=args.output_dir)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ------
    # model
    # ------
    backbone = ConvNet(args.hidden_dim, args.context_dims, args.pyramid_dims, args.dropout)
    # backbone = GatedNet(args.hidden_dim, args.context_dims, args.dropout)
    model = Classifier(args, backbone, val_names, test_names)

    # ---------
    # training
    # ---------
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         deterministic=True,
                         gradient_clip_val=args.grad_clip_val,
                         accelerator='ddp', gpus=args.num_gpus,
                         logger=logger, callbacks=[lr_monitor],
                         num_sanity_val_steps=0,
                         plugins=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(f"lightning_logs/{args.output_dir}.ckpt")

    # -----------
    # finetuning
    # -----------
    if args.finetune==True:
        # -------
        # logger
        # -------
        args.lr_init = lr_monitor.lrs[-1]
        logger = TensorBoardLogger('lightning_logs', name=f'{args.output_dir}_tune')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # ------
        # model
        # ------
        backbone = ConvNet(args.hidden_dim, args.context_dims, args.pyramid_dims, args.dropout)
        model = Classifier.load_from_checkpoint(f"lightning_logs/{args.output_dir}.ckpt",
                                                hparams=args, backbone=backbone,
                                                val_names=val_names, test_names=test_names)

        # ---------
        # training
        # ---------
        trainer = pl.Trainer(max_epochs=args.tune_epochs,
                             deterministic=True,
                             gradient_clip_val=args.grad_clip_val,
                             accelerator='ddp', gpus=args.num_gpus,
                             logger=logger, callbacks=[lr_monitor],
                             num_sanity_val_steps=0,
                             plugins=DDPPlugin(find_unused_parameters=False))
        trainer.fit(model, tune_loader, val_loader)
        trainer.save_checkpoint(f"lightning_logs/{args.output_dir}_tune.ckpt")

    # --------
    # testing
    # --------
    trainer.test(test_dataloaders=test_loader)
