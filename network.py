import torch
import optuna
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
import pickle

from torch.utils.data                          import DataLoader
from pytorch_lightning.loggers                 import TensorBoardLogger
from pytorch_lightning.callbacks               import LearningRateMonitor
from pytorch_lightning.plugins                 import DDPPlugin
from pytorch_lightning.metrics.classification  import PrecisionRecallCurve as PRCurve
from pytorch_lightning.metrics.functional      import auc, precision_recall_curve as prc, roc
from optuna.integration                        import PyTorchLightningPruningCallback


class Dataset(torch.utils.data.Dataset):
    """Dataset class for generating/loading batches"""
    def __init__(self, root_dir, rel_path, context_dims, min_cells, batchSize=None, wShuffle=0.05,
                 max_lag=0.05, overwrite=False, load_prev=True, verbose=False):

        self.root_dir = root_dir
        self.rel_path = rel_path
        self.context_dims = context_dims
        self.batch_size = batchSize
        self.overwrite = overwrite
        self.load_prev = load_prev
        self.pt_shuffle = wShuffle
        self.min_cells = min_cells
        self.max_lag = max_lag

        if self.load_prev==True:
            # ---------------------------
            # get batch pathnames (X, y)
            # ---------------------------
            prev_path = '/'.join(self.rel_path.split('/')[:-1])+'*/'
            self.X_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'X_*.npy'))]
            self.y_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'y_b*.npy'))]

        else:
            # ----------------------------------------------
            # generate batch(es) for each trajectory (X, y)
            # ----------------------------------------------
            self.sce_fnames = sorted(pathlib.Path(self.root_dir).glob(self.rel_path))
            self.X_fnames = [None] * len(self.sce_fnames)
            self.y_fnames = [None] * len(self.sce_fnames)
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

            # shuffle the cells in pseudotime on each batch
            order = self.shuffle_pt(np.arange(X.shape[-1]),
                        random.randint(0,1000000), False)

            # add uniform noise to each batch
            X += np.random.uniform(-1e-4, 1e-4, size=X.shape).astype(np.float32)

            if X.shape[3] > self.min_cells : new_batch = False
            idx_ = np.random.choice(np.arange(len(self.X_fnames)))
        return X[:,:,:,order], y, fname


    def seed_from_string(self, s):
        """Generate random seed given a string"""
        n = int.from_bytes(s.encode(), 'little')
        return sum([int(x) for x in str(n)])


    def grouper(self, iterable, m, fillvalue=None):
        """Split iterable into chunks of size m"""
        args = [iter(iterable)] * m
        return itertools.zip_longest(*args, fillvalue=fillvalue)


    def shuffle_pt(self, pt, seed, df=True):
        """Kernelized swapper"""
        np.random.seed(seed)
        if df==True: pt = pt.copy().values
        for i in np.arange(pt.size):
            j = np.random.normal(loc=0, scale=self.pt_shuffle*pt.size)
            i_ = int(round(np.clip(i+j, 0, pt.size-1)))
            pt[[i,i_]] = pt[[i_,i]]
        return pt


    def max_cross_correlation(self, a, v):
        corr = abs(np.correlate(a, v, "same"))
        lag_min = corr.size//2 - corr.size//int(1/self.max_lag)
        lag_max = corr.size//2 + corr.size//int(1/self.max_lag)
        return corr[ lag_min : lag_max ].max()


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

                # print message if generating batches for experimental dataset
                if sce_folder.split('/')[-4]=='experimental':
                    print(f"Generating batches for {'/'.join(sce_folder.split('/')[-2:])}...")

                if sce is not None: pass
                else:
                    # load single cell experiment data from file
                    sce = pd.read_csv(sce_fname, index_col=0).T

                    # lowercase gene names
                    sce.columns = sce.columns.str.lower()

                    # per gene normalization (vector norm)
                    for col in sce.columns:
                        sce[col] /= np.sqrt((sce[col]**2).sum())

                # get trajectory indices, sorted by slingshot pseudotime values
                traj_idx = pt.iloc[np.where(~pt.iloc[:,k].isnull())[0], k].sort_values().index

                # trajectory pairwise gene correlations: max absolute cross correlation
                if traj_idx.size > self.min_cells:
                    traj_pcorr = sce.loc[traj_idx,:].corr(self.max_cross_correlation)
                else: traj_pcorr = sce.corr(self.max_cross_correlation)

                # trajectory pseudotime (sorted) -> pseudotime step (normalized)
                pt_k = pt.loc[traj_idx, pt.columns[k]].values
                pt_k_dt = np.concatenate((np.zeros((1,)), pt_k[1:]-pt_k[0:-1]))
                pt_k_dt /= np.sqrt((pt_k_dt**2).sum())

                # if synthetic experiment, shuffle pseudotime
                if sce_folder.split('/')[-4]!='experimental':
                    seed = self.seed_from_string(traj_folder)
                # NOTE: will only shuffle expression, not pseudotime
                    traj_idx = self.shuffle_pt(traj_idx, seed)

                if ref is not None: pass
                else:
                    # load gene regulation labels from reference file
                    ref = pd.read_csv(f"{sce_folder}/refNetwork.csv")
                    g1 = [g.lower() for g in ref['Gene1'].values]
                    g2 = [g.lower() for g in ref['Gene2'].values]
                    ref_1d = np.array(["%s %s" % x for x in list(zip(g1,g2))])

                # generate list of tuples containing all possible gene pairs & context (gene A, context are tfs)
                gpairs = [tuple(list(g)+list(traj_pcorr.loc[g[0],(traj_pcorr.index.isin(g1))&(~traj_pcorr.index.isin(g))].nlargest(self.context_dims).index)
                                        +list(traj_pcorr.loc[g[1],(traj_pcorr.index.isin(g1))&(~traj_pcorr.index.isin(g))].nlargest(self.context_dims).index))
                                                                                                            for g in itertools.product(set(g1), sce.columns)]
                seed = self.seed_from_string(traj_folder); random.seed(seed); random.shuffle(gpairs)

                n, n_cells = len(gpairs), traj_idx.size

                # split gene pairs & context into batches
                if self.batch_size is not None:
                    gpairs_batched = [list(x) for x in self.grouper(gpairs, self.batch_size)]
                    gpairs_batched = [list(filter(None, x)) for x in gpairs_batched]
                else: gpairs_batched = [gpairs]

                # inner loop over batches of gene pairs & context
                for j in range(len(gpairs_batched)):
                    X_fname = f"{traj_folder}/X_batch{j}_size{len(gpairs_batched[j])}.npy"
                    y_fname = f"{traj_folder}/y_batch{j}_size{len(gpairs_batched[j])}.npy"

                    # flatten gene pairs & context (list of tuples) to list
                    gpairs_list = list(itertools.chain(*gpairs_batched[j]))

                    # split minibatch into single examples of gene A -> geneB & context
                    if self.batch_size is None or j==len(gpairs_batched)-1:
                        sce_list = np.array_split(sce.loc[traj_idx, gpairs_list].values, len(gpairs_batched[j]), axis=1)
                    else:
                        sce_list = np.array_split(sce.loc[traj_idx, gpairs_list].values, self.batch_size, axis=1)

                    # recombine re-shaped examples into full mini-batch
                    sce_list = [g_sce.reshape(1,2+self.context_dims*2,1,n_cells) for g_sce in sce_list]
                    X_batch = np.concatenate(sce_list, axis=0).astype(np.float32)

                    # generate gene regulation labels y for mini-batch
                    gpairs_batched_1d = np.array(["%s %s" % x[:2] for x in gpairs_batched[j]])
                    y_batch = np.in1d(gpairs_batched_1d, ref_1d).reshape(X_batch.shape[0],1)

                    # concat pseudotime step to each example in mini-batch
                    pt_ = np.broadcast_to(pt_k_dt, (X_batch.shape[0], 1, 1, pt_k.shape[0]))
                    X_batch = np.concatenate((pt_, X_batch), axis=1).astype(np.float32)
                    y_float = y_batch.astype(np.float32)

                    # save X, y to pickled numpy files
                    np.save(X_fname, X_batch, allow_pickle=True)
                    np.save(y_fname, y_float, allow_pickle=True)

                    # save batch filenames for __len__ and __getitem__
                    idx = np.where(np.array(self.X_fnames) == None)[0]
                    if idx.size > 0:
                        self.X_fnames[idx[0]] = X_fname
                        self.y_fnames[idx[0]] = y_fname
                    else:
                        self.X_fnames.extend([X_fname])
                        self.y_fnames.extend([y_fname])


class BranchedBlock(torch.nn.Module):
    def __init__(self, in_channels,
                 hidden_dim, filters,
                 transition=False,
                 out_channels=None,
                 maxpool=True,
                 batchnorm=True,
                 attention=True,
                 skip_connection=True,
                 negative_slope=0.2):

        super(BranchedBlock, self).__init__()
        self.do_maxpool = maxpool
        self.do_transition = transition
        self.do_batchnorm = batchnorm
        self.attention = attention
        self.skip = skip_connection
        Activation = torch.nn.LeakyReLU

        # -----------------------
        # convolutional branches
        # -----------------------
        self.branches = torch.nn.ModuleList([ torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=(1,filters[i]), padding=(0,filters[i]//2)),
            Activation(negative_slope=negative_slope)) for i in range(len(filters)) ])
        for i in range(len(self.branches)): self.branches[i].apply(init_weights_scaled)

        # --------------
        # maxpool layer
        # --------------
        if self.do_maxpool==True: self.maxpool = torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,1), padding=(0,1))

        # ---------------
        # 1x1 transition
        # ---------------
        if self.do_transition==True and out_channels is not None:
            self.transition_1x1 = torch.nn.Sequential(
                torch.nn.Conv2d(len(self.branches)*hidden_dim, out_channels, kernel_size=(1,1)),
                Activation(negative_slope=negative_slope))
            self.transition_1x1.apply(init_weights_scaled)

        # ----------------
        # batchnorm layer
        # ----------------
        if self.do_batchnorm==True: self.batchnorm = torch.nn.BatchNorm2d(
            out_channels if self.do_transition==True and out_channels is not None else len(self.branches)*hidden_dim)

        # ------------------
        # channel attention
        # ------------------
        if self.attention==True: self.channel_attention = ChannelAttentionLayer(
            out_channels if self.do_transition==True and out_channels is not None else len(self.branches)*hidden_dim)

    # --------
    # forward
    # --------
    def forward(self, x, p):

        # -----------------------
        # convolutional branches
        # -----------------------
        out = self.branches[0](x)
        for i in range(1,len(self.branches)):
            br_out = self.branches[i](x)
            out = torch.cat([out, br_out], axis=1)

        # --------------
        # maxpool layer
        # --------------
        if self.do_maxpool==True:
            out = self.maxpool(out)

        # -----------------
        # transition layer
        # -----------------
        if self.do_transition==True:
            out = self.transition_1x1(out)

        # --------------------
        # batchnorm + dropout
        # --------------------
        if self.do_batchnorm==True:
            out = self.batchnorm(out)
        out = F.dropout(out, p)

        # ------------------
        # channel attention
        # ------------------
        if self.attention==True:
            out = self.channel_attention(out)

        # ----------------
        # skip connection
        # ----------------
        if self.skip==True:
            out += x
        return out


class ConvNet(torch.nn.Module):
    def __init__(self, num_layers,
                 filters, hidden_dim,
                 context_dims,
                 negative_slope=0.2):

        super(ConvNet, self).__init__()
        self.d_context = context_dims
        self.d_hidden = hidden_dim
        self.n_layers = num_layers
        Activation = torch.nn.LeakyReLU

        # -----------
        # pseudotime
        # -----------
        self.time_block = BranchedBlock(in_channels=1,
                                        hidden_dim=self.d_hidden,
                                        filters=filters,
                                        transition=True,
                                        out_channels=1,
                                        maxpool=False,
                                        batchnorm=False,
                                        attention=False,
                                        skip_connection=False)

        # # ---------------------
        # # single-gene features
        # # ---------------------
        # self.single_gene_blocks = torch.nn.ModuleList([BranchedBlock(in_channels=2,
        #                                                              hidden_dim=self.d_hidden,
        #                                                              filters=filters,
        #                                                              transition=True,
        #                                                              out_channels=1,
        #                                                              maxpool=False,
        #                                                              batchnorm=False,
        #                                                              attention=False,
        #                                                              skip_connection=False)
        #                                                for i in range(2+2*self.d_context)])

        # # -------------------
        # # gene-pair features
        # # -------------------
        # self.gene_pair_blocks = torch.nn.ModuleList([BranchedBlock(in_channels=3,
        #                                                            hidden_dim=self.d_hidden,
        #                                                            filters=filters,
        #                                                            transition=True,
        #                                                            out_channels=1,
        #                                                            maxpool=False,
        #                                                            batchnorm=False,
        #                                                            skip_connection=False)
        #                                              for i in range(2*(2+2*self.d_context))])

        # ------------------
        # all-gene features
        # ------------------
        self.core_blocks = torch.nn.ModuleList([BranchedBlock(in_channels=3,
                                                              hidden_dim=self.d_hidden,
                                                              filters=filters,
                                                              transition=False,
                                                              maxpool=True,
                                                              batchnorm=True,
                                                              attention=True,
                                                              skip_connection=False)])

        self.core_blocks.extend([BranchedBlock(in_channels=len(filters)*self.d_hidden,
                                               hidden_dim=self.d_hidden, filters=filters,
                                               transition=False,
                                               maxpool=True,
                                               batchnorm=True,
                                               attention=True,
                                               skip_connection=True)
                                               for i in range(1,self.n_layers)])

        # -----------------
        # adaptive pooling
        # -----------------
        self.max_pooling = torch.nn.AdaptiveMaxPool2d(1)

        # ----------------------
        # fully connected layer
        # ----------------------
        self.linear_output = torch.nn.Linear(len(filters)*self.d_hidden, 1)
        self.linear_output.apply(init_weights_scaled)

    # ---------
    # forward
    # ---------
    def forward(self, x, p_block, p_final):
        dt, out = x[:,0,:,:], x[:,1:3,:,:]

        # -----------
        # pseudotime
        # -----------
        dt = self.time_block(dt.unsqueeze(2), p_block)

        # # ---------------------
        # # single-gene features
        # # ---------------------
        # for i in range(out.size(1)):
        #     bl_in = torch.cat([dt, out[:,i,:,:].unsqueeze(2)], axis=1)
        #     bl_out = self.single_gene_blocks[i](bl_in, p_block)
        #     out[:,i,:,:] = torch.squeeze(bl_out).unsqueeze(1)

        # # -------------------
        # # gene-pair features
        # # -------------------
        # features = torch.clone(dt)
        # for i in range(out.size(1)):
        #     # gene pairs with gene A
        #     bl_in = torch.cat([dt, out[:,0,:,:].unsqueeze(2), out[:,i,:,:].unsqueeze(2)], axis=1)
        #     bl_out = self.gene_pair_blocks[(2*i)](bl_in, p_block)
        #     features = torch.cat([features,bl_out], axis=1)
        #
        #     # gene pairs with gene B
        #     bl_in = torch.cat([dt, out[:,1,:,:].unsqueeze(2), out[:,i,:,:].unsqueeze(2)], axis=1)
        #     bl_out = self.gene_pair_blocks[(2*i)+1](bl_in, p_block)
        #     features = torch.cat([features,bl_out], axis=1)
        # out = torch.clone(features)

        # ------------------
        # all-gene features
        # ------------------
        out = torch.cat([dt,out], axis=1)
        for i in range(self.n_layers):
            out = self.core_blocks[i](out, p_block)

        # -----------------
        # adaptive pooling
        # -----------------
        # out = torch.cat([out1,out2], axis=1)
        out = self.max_pooling(out)

        # ----------------------
        # fully connected layer
        # ----------------------
        out = torch.flatten(out, start_dim=1)
        out = F.dropout(out, p_final)
        return self.linear_output(out)


class ChannelAttentionLayer(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionLayer, self).__init__()

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.conv_du = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels//reduction, 1, padding=0, bias=True), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels//reduction, channels, 1, padding=0, bias=True), torch.nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Classifier(pl.LightningModule):
    """Deep neural network for binary classification of gene trajectory pairs"""
    def __init__(self, hparams, backbone, val_names):
    # def __init__(self, hparams, backbone, val_names, test_names):

        super().__init__()
        self.save_hyperparameters(hparams)

        self.backbone = backbone

        # -------------------------------------------------
        # store preds & targets in precision-recall curves
        # -------------------------------------------------
        self.val_names = val_names
        # self.test_names = test_names
        self.val_prc = torch.nn.ModuleList([PRCurve(pos_label=1) for x in self.val_names])
        # self.test_prc = torch.nn.ModuleList([PRCurve(pos_label=1) for x in self.test_names])

    # -------------------------
    # optimizer & LR scheduler
    # -------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr_init, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.hparams.patience, factor=0.1, min_lr=0.001*self.hparams.lr_init)
        return { 'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler, 'monitor' : 'val_loss_exp' }

    # -------------
    # forward pass
    # -------------
    def forward(self, x, p_block, p_final):
        return self.backbone(x, p_block, p_final)

    # --------------
    # training step
    # --------------
    def training_step(self, train_batch, batch_idx):
        """Optimize mean loss per batch"""
        X, y, fname = train_batch
        p_block = self.hparams.dropout_block
        p_final = self.hparams.dropout_final
        out = self.forward(X, p_block, p_final)
        pred = torch.sigmoid(out)
        loss = F.binary_cross_entropy_with_logits(out, y, weight=y.sum()/y.size(0), reduction='sum')/self.hparams.batch_size

        # self.logger.experiment.add_histogram('train_pred', pred, global_step=self.global_step)
        # self.logger.experiment.add_histogram('fc_weights', fc_w, global_step=self.global_step)
        # self.logger.experiment.add_histogram('conv_weights', conv_w, global_step=self.global_step)

        # ------------
        # update logs
        # ------------
        if fname.split('/')[-6]=='experimental':
            self.log('train_loss_exp', loss,
                     on_step=True,
                     on_epoch=True,
                     sync_dist=True)
        else:
            self.log('train_loss_synt', loss,
                     on_step=True,
                     on_epoch=True,
                     sync_dist=True)

        return loss

    # ----------------
    # validation step
    # ----------------
    def validation_step(self, val_batch, batch_idx, dataset_idx) -> None:
        """Val step updates metrics for each model across all datasets"""
        X, y, fname = val_batch
        p_block = self.hparams.dropout_block
        p_final = self.hparams.dropout_final
        out = self.forward(X, p_block, p_final)
        pred = torch.sigmoid(out)
        loss = F.binary_cross_entropy_with_logits(out, y, weight=y.sum()/y.size(0), reduction='sum')/self.hparams.batch_size

        # ----------------
        # update PR curve
        # ----------------
        self.val_prc[dataset_idx](pred, y)

        # ------------
        # update logs
        # ------------
        if fname.split('/')[-6]=='experimental':
            self.log('val_loss_exp', loss,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True,
                     add_dataloader_idx=False)
        else:
            self.log('val_loss_synt', loss,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True,
                     add_dataloader_idx=False)

    # ---------------------
    # validation epoch end
    # ---------------------
    def on_validation_epoch_end(self):
        """Compute AUPRC & AUROC for each model/experiment"""
        val_auprc = torch.zeros((len(self.val_names),), device=torch.cuda.current_device())
        val_auroc = torch.zeros((len(self.val_names),), device=torch.cuda.current_device())
        for idx in range(len(self.val_names)):
            name = self.val_names[idx]
            # NOTE: currently, AUC averaged across processes
            preds = torch.cat(self.val_prc[idx].preds, dim=0)
            target = torch.cat(self.val_prc[idx].target, dim=0)
            precision, recall, _ = prc(preds, target, pos_label=1)
            fpr, tpr, _ = roc(preds, target, pos_label=1)
            auprc, auroc = auc(recall, precision), auc(fpr, tpr)
            val_auprc[idx] = auprc; val_auroc[idx] = auroc
            self.log(f'{name}_auprc', auprc, sync_dist=True,
                     add_dataloader_idx=False)
            self.log(f'{name}_auroc', auroc, sync_dist=True,
                     add_dataloader_idx=False)
            self.val_prc[idx].reset()
        if val_auroc.mean() == 0.5: mean_auroc = 0.
        else: mean_auroc = val_auroc.mean()
        self.log(f'val_avg_auroc', mean_auroc, sync_dist=True,
                 add_dataloader_idx=False)
        self.log(f'val_avg_auprc', val_auprc.mean(), sync_dist=True,
                 add_dataloader_idx=False)

    # # ----------
    # # test step
    # # ----------
    # def test_step(self, test_batch, batch_idx, dataset_idx) -> None:
    #     """Test step used to update metrics on individual datasets"""
    #     X, y, fname = test_batch
    #     out = self.forward(X)
    #     pred = torch.sigmoid(out)
    #
    #     # ----------------
    #     # update PR curve
    #     # ----------------
    #     self.test_prc[dataset_idx](pred, y)

    # # ---------------
    # # test epoch end
    # # ---------------
    # def on_test_epoch_end(self):
    #     """Compute AUPRC & AUROC for each dataset"""
    #     for idx in range(len(self.test_names)):
    #         name = self.test_names[idx]
    #         # NOTE: currently, AUC averaged across processes
    #         preds = torch.cat(self.test_prc[idx].preds, dim=0)
    #         target = torch.cat(self.test_prc[idx].target, dim=0)
    #         precision, recall, _ = prc(preds, target, pos_label=1)
    #         fpr, tpr, _ = roc(preds, target, pos_label=1)
    #         auprc, auroc = auc(recall, precision), auc(fpr, tpr)
    #         self.log(f'{name}_auprc', auprc, sync_dist=True,
    #                  add_dataloader_idx=False)
    #         self.log(f'{name}_auroc', auroc, sync_dist=True,
    #                  add_dataloader_idx=False)
    #         self.test_prc[idx].reset()


# class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
#
#     def __init__(self, optimizer, warmup, max_iters):
#         self.warmup = warmup
#         self.max_num_iters = max_iters
#         super().__init__(optimizer)
#
#     def get_lr(self):
#         lr_factor = self.get_lr_factor(epoch=self.last_epoch)
#         return [base_lr * lr_factor for base_lr in self.base_lrs]
#
#     def get_lr_factor(self, epoch):
#         lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
#         if epoch <= self.warmup:
#             lr_factor *= epoch * 1.0 / self.warmup
#         return lr_factor


class Objective(object):
    """Optuna objective for efficient hyperparameter search and pruning"""
    def __init__(self, args, train_loader, val_loader, val_names, monitor):

        # -----
        # args
        # -----
        self.args = args

        # -----
        # data
        # -----
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_names = val_names

        # ------------
        # monitor val
        # ------------
        self.monitor = monitor


    def __call__(self, trial):

        # ----------------
        # suggest hparams
        # ----------------
        self.args.lr_init = trial.suggest_float("lr_init", 1e-5, 1e-2, log=True)
        self.args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 10, log=True)
        # self.args.gamma = trial.suggest_float("gamma", 0.1, 10., log=True)
        # self.args.hidden_dim = trial.suggest_int("hidden_dim", 4, 64, step=4)
        # self.args.tf_heads = trial.suggest_int("tf_heads", 2, 4, step=2)

        # -------
        # logger
        # -------
        logger = TensorBoardLogger('optuna_logs', name=f'{self.args.output_dir}_trial{trial.number}', version=0)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # -----------
        # calllbacks
        # -----------
        early_stop_metrics = pl.callbacks.EarlyStopping(self.monitor, patience=self.args.patience, mode='max')
        early_stop_train = pl.callbacks.EarlyStopping('train_loss_exp', patience=self.args.patience, mode='min')
        early_stop_val = pl.callbacks.EarlyStopping('val_loss_exp', patience=self.args.patience, mode='min')
        checkpoint_callback = pl.callbacks.ModelCheckpoint('optuna_logs', f"{self.args.output_dir}_trial{trial.number}.ckpt")

        # ------
        # model
        # ------
        backbone = ConvNet(self.args.hidden_dim,
                           self.args.tf_heads,
                           self.args.tf_layers,
                           self.args.attn_size,
                           self.args.conv_size,
                           args.len_train_X,
                           self.args.context_dims,
                           self.args.pyramid_dims,
                           self.args.dropout)
        model = Classifier(self.args, backbone, self.val_names)

        # ---------
        # training
        # ---------
        trainer = pl.Trainer(max_epochs=self.args.max_epochs, limit_train_batches=self.args.train_frac,
                             deterministic=True, accelerator='ddp_spawn', gpus=self.args.num_gpus, logger=logger,
                             checkpoint_callback=checkpoint_callback, num_sanity_val_steps=0,
                             plugins=[DDPPlugin(find_unused_parameters=False)], callbacks=[lr_monitor, early_stop_metrics,
                             early_stop_train, early_stop_val, PyTorchLightningPruningCallback(trial, self.monitor)])
        trainer.fit(model, self.train_loader, self.val_loader)
        trainer.save_checkpoint(f"optuna_logs/{self.args.output_dir}_trial{trial.number}.ckpt")

        return trainer.callback_metrics[self.monitor]


def init_weights_scaled(m):
    if type(m) in [torch.nn.Conv2d, torch.nn.Linear]:
        torch.nn.init.kaiming_uniform_(m.weight)


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
    parser.add_argument('--load_prev', type=get_bool, default=True)
    parser.add_argument('--ovr_train', type=get_bool, default=False)
    parser.add_argument('--ovr_val', type=get_bool, default=False)
    # parser.add_argument('--ovr_test', type=get_bool, default=False)
    parser.add_argument('--ovr_tune', type=get_bool, default=False)
    parser.add_argument('--train_frac', type=float, default=.33)
    parser.add_argument('--tune_hparams', type=get_bool, default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--finetune', type=get_bool, default=False)
    parser.add_argument('--tune_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--context_dims', type=int, default=2)
    parser.add_argument('--min_cells', type=int, default=50)
    parser.add_argument('--lr_init', type=float, default=.00001)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--filters', type=str, default='1,7,31')
    parser.add_argument('--hidden_dim', type=int, default=12)
    parser.add_argument('--dropout_block', type=float, default=0.)
    parser.add_argument('--dropout_final', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--grad_clip_val', type=float, default=.01)
    parser.add_argument('--num_workers', type=int, default=36)
    parser.add_argument('--num_gpus', type=int, default=2)
    args = parser.parse_args()

    args.filters = [int(x) for x in args.filters.split(',')]

    if args.tune_hparams==True:
        args.num_workers = 0

    # -----------------
    # training dataset
    # -----------------
    print('Loading training batches...')
    training = Dataset(root_dir=f'{args.dataset_dir}/training', rel_path='*/*/*/*/ExpressionData.csv',
                       context_dims=args.context_dims, min_cells=args.min_cells, batchSize=args.batch_size,
                       overwrite=args.ovr_train, load_prev=args.load_prev, verbose=True)
    train_loader = DataLoader(training, batch_size=None, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # -------------
    # tune dataset
    # -------------
    if args.finetune==True:
        print('Loading fine-tune batches...')
        training_tune = Dataset(root_dir=f'{args.dataset_dir}/training_tune', rel_path='*/*/*/*/ExpressionData.csv',
                                context_dims=args.context_dims, min_cells=args.min_cells, batchSize=args.batch_size,
                                overwrite=args.ovr_tune, load_prev=args.load_prev, verbose=True)
        tune_loader = DataLoader(training_tune, batch_size=None, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # --------------------
    # validation datasets
    # --------------------
    validation, val_names = [], []
    print('Loading validation datasets...')
    for item in tqdm(sorted(glob.glob(f'{args.dataset_dir}/validation/*/*/*'))):
        if os.path.isdir(item):
            val_names.append('val_'+'_'.join(item.split('/')[-2:]))
            validation.append(Dataset(root_dir=item, rel_path='*/ExpressionData.csv', context_dims=args.context_dims,
            min_cells=args.min_cells, batchSize=args.batch_size, overwrite=args.ovr_val, load_prev=args.load_prev))

    val_loader = [None] * len(validation)
    for i in range(len(validation)):
        val_loader[i] = DataLoader(validation[i], batch_size=None, num_workers=args.num_workers, pin_memory=True)

    # # -----------------
    # # testing datasets
    # # -----------------
    # testing, test_names = [], []
    # print('Loading testing datasets...')
    # for item in tqdm(sorted(glob.glob(f'{args.dataset_dir}/testing/*/*/*/*'))):
    #     if os.path.isdir(item):
    #         test_names.append('test_'+'_'.join(item.split('/')[-2:]))
    #         testing.append(Dataset(root_dir=item, rel_path='ExpressionData.csv',
    #         context_dims=args.context_dims, batchSize=args.batch_size,
    #         overwrite=args.ovr_val, load_prev=args.load_prev))
    #
    # test_loader = [None] * len(testing)
    # for i in range(len(testing)):
    #     test_loader[i] = DataLoader(testing[i], batch_size=None, num_workers=args.num_workers, pin_memory=True)

    if args.load_prev==False:
        input('Completed overwrite.')

    if args.tune_hparams==True:
        # --------------
        # hparam tuning
        # --------------
        sampler = optuna.samplers.TPESampler(seed=1234)
        pruner =  optuna.pruners.ThresholdPruner(lower=0.45, n_warmup_steps=args.patience) ## need parameter for threshold
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(Objective(args, train_loader, val_loader, val_names,
                                 monitor='val_hHep_camp-et-al-2017_avg_auc'), n_trials=100)
        pickle.dump(study, open(f'optuna_logs/{args.output_dir}_study.pkl', 'wb'))

    else:
        # -------
        # logger
        # -------
        logger = TensorBoardLogger('lightning_logs', name=args.output_dir)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # ------
        # model
        # ------
        backbone = ConvNet(args.num_layers,
                           args.filters,
                           args.hidden_dim,
                           args.context_dims)
        model = Classifier(args, backbone, val_names)
        # model = Classifier(args, backbone, val_names, test_names)

        # ---------
        # training
        # ---------
        trainer = pl.Trainer(max_epochs=args.max_epochs, deterministic=True, accelerator='ddp',
                             gpus=[0,],
                             # gpus=args.num_gpus,
                             logger=logger, callbacks=[lr_monitor],
                             num_sanity_val_steps=0, plugins=[DDPPlugin(find_unused_parameters=False)],
                             limit_train_batches=args.train_frac, gradient_clip_val=args.grad_clip_val)
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
            backbone = ConvNet(args.num_layers,
                               args.hidden_dim,
                               args.context_dims,
                               args.pyramid_dims,
                               args.dropout)
            model = Classifier.load_from_checkpoint(f"lightning_logs/{args.output_dir}.ckpt",
                                                    hparams=args, backbone=backbone, val_names=val_names)

            # ---------
            # training
            # ---------
            trainer = pl.Trainer(max_epochs=args.tune_epochs, deterministic=True, accelerator='ddp',
                                 # gpus=args.num_gpus,
                                 gpus=[0,],
                                 logger=logger, callbacks=[lr_monitor],
                                 num_sanity_val_steps=0, plugins=[DDPPlugin(find_unused_parameters=False)],
                                 limit_train_batches=args.train_frac, gradient_clip_val=args.grad_clip_val)
            trainer.fit(model, tune_loader, val_loader)
            trainer.save_checkpoint(f"lightning_logs/{args.output_dir}_tune.ckpt")

    # # --------
    # # testing
    # # --------
    # trainer.test(test_dataloaders=test_loader)
