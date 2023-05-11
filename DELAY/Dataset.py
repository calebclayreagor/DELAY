import argparse
import torch
import os
import random
import itertools
import numpy as np
import pandas as pd
from typing import Tuple
from typing import List
from typing import TypeVar
from pathlib import Path
from tqdm import tqdm

Self = TypeVar('Self', bound = 'Dataset')

class Dataset(torch.utils.data.Dataset):
    """Load or compile mini-batches of joint-probability matrices for the given dataset"""

    def __init__(self: Self, 
                 args: argparse.Namespace, 
                 ds_dir: str,
                 split: str) -> Self:

        self.args = args
        self.ds_dir = ds_dir
        self.split = split

        # get filenames for mini-batches
        self.outdir = f'{ds_dir}{split}'
        if os.path.isdir(self.outdir):
            self.X_fn = list(map(str, sorted(Path(self.ds_dir).glob(f'{split}/X_*.npy'))))
            self.y_fn = list(map(str, sorted(Path(self.ds_dir).glob(f'{split}/y_*.npy'))))
            self.msk_fn = list(map(str, sorted(Path(self.ds_dir).glob(f'{split}/msk_*.npy'))))
            print(f'Loaded existing batches for {"/".join(self.outdir.split("/")[-2:])}...')
   
        else: 
            # compile mini-batches
            os.mkdir(self.outdir)
            self.compile_batches()

    def __len__(self: Self) -> int:
        return len(self.X_fn)

    def __getitem__(self: Self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        X = np.load(self.X_fn[idx])
        y = np.load(self.y_fn[idx])
        msk = np.load(self.msk_fn[idx])
        return X, y, msk, self.X_fn[idx].split('X_')

    def shuffle_pseudotime(self: Self, pt: np.ndarray) -> np.ndarray:
        """Probabilistically swap cells' positions along trajectory at the given scale"""
        np.random.seed(pt.size)
        for i in np.arange(pt.size):
            j = np.random.normal(loc = 0, scale = self.args.shuffle * pt.size)
            _i_ = int(round(np.clip(i + j, 0, pt.size - 1)))
            pt[[i, _i_]] = pt[[_i_, i]]
        return pt

    def compile_batches(self: Self) -> None:
        """Use sce to compile mini-batches for dataset and save as .npy files"""
        # load normalized data and PseudoTime values from sce dataset
        ds = pd.read_csv(f'{self.ds_dir}NormalizedData.csv', index_col = 0).T
        ds.columns = ds.columns.str.lower() # gene names in lowercase
        pt = pd.read_csv(f'{self.ds_dir}PseudoTime.csv', index_col = 0)
        print(f'Compiling batches for {"/".join(self.outdir.split("/")[-2:])}...')

        # load gene-regulation pairs from refNetwork file
        ref_network = pd.read_csv(f'{self.ds_dir}refNetwork.csv')
        ref_network['Gene1'] = ref_network['Gene1'].str.lower()
        ref_network['Gene2'] = ref_network['Gene2'].str.lower()
        g1 = list(ref_network['Gene1'].values)
        g2 = list(ref_network['Gene2'].values)
        ref_1d = np.array([f'{g[0]} {g[1]}' for g in list(zip(g1, g2))])

        # list TFs from refNetwork or TranscriptionFactor file
        if self.args.predict == True:
            tf = np.loadtxt(f'{self.ds_dir}TranscriptionFactors.csv', delimiter=',', dtype=str)
            tf = list(np.char.lower(tf))
        else: tf = g1.copy()

        # choose appropriate TFs to use as Gene1 if cross-validating or predicting
        if self.args.valsplit is not None:
            labels = pd.read_csv(f'{self.ds_dir}splitLabels.csv', index_col = 0)
            labels.index = labels.index.str.lower()
            if self.split == 'training': 
                g1 = list(labels[(labels.values != self.args.valsplit)].index)
            elif self.split == 'validation':
                g1 = list(labels[(labels.values == self.args.valsplit)].index)
        elif self.split == 'prediction': g1 = tf.copy()

        # sort single cells by pseudotime values
        cell_idx = pt.sort_values('PseudoTime').index.values

        # shuffle pseudotime values (optional)
        if self.args.shuffle is not None:
            cell_idx = self.shuffle_pseudotime(cell_idx)

        # sample ncells [optional] (only if len(traj) > ncells)
        if self.args.ncells is not None and self.args.ncells < cell_idx.size:
            idx_select = np.arange(cell_idx.size)
            idx_select = np.random.choice(idx_select, self.args.ncells, False)
            cell_idx = cell_idx[np.sort(idx_select)]

        # additional sequencing dropouts (optional)
        if self.args.dropout is not None:
            below_cutoff = ds.loc[cell_idx,:].values < ds.loc[cell_idx,:].quantile(self.args.dropout, axis=1).values[:,None]
            drop_indicator = np.random.choice([0,1], p = [self.args.dropout, 1 - self.args.dropout], size = below_cutoff.shape)
            ds.loc[cell_idx,:] *= (below_cutoff.astype(int) * drop_indicator + (~below_cutoff).astype(int))

        # gene pairwise correlations: max absolute cross correlation or max pearson correlation
        if self.args.max_lag > 0: 
            gpcorr = ds.loc[cell_idx,:].corr(lambda a,b : abs(np.correlate(a[:-self.args.max_lag], b)).max())
        else: 
            gpcorr = ds.loc[cell_idx,:].corr(method = 'pearson')
        
        # select gene pairs in given regulatory motif; ablate/mask neighbor genes in motif (both optional)
        gmasks, gpair_select = dict(), np.array([''])
        for g in itertools.product(sorted(set(g1)), ds.columns):
            if self.args.motif is None:
                gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                gmasks[g] = np.ones((ds.shape[1],)).astype(bool)

            # feed-forward loop or shared upstream regulator
            elif self.args.motif == 'ffl-reg':
                pair_reg = ref_network.loc[ref_network['Gene2'].isin(g), 'Gene1']
                pair_reg = pair_reg[pair_reg.duplicated()].values
                if pair_reg.size > 0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate == True: gmasks[g] = ~ds.columns.isin(pair_reg)
                else: gmasks[g] = np.ones((ds.shape[1],)).astype(bool)

            # feed-forward loop or shared downstream target
            elif self.args.motif == 'ffl-tgt':
                pair_tgt = ref_network.loc[ref_network['Gene1'].isin(g), 'Gene2']
                pair_tgt = pair_tgt[pair_tgt.duplicated()].values
                if pair_tgt.size > 0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate == True: gmasks[g] = ~ds.columns.isin(pair_tgt)
                else: gmasks[g] = np.ones((ds.shape[1],)).astype(bool)

            # feed-forward loop or transitive (sequential) regulation
            elif self.args.motif == 'ffl-trans':
                pair_trans = np.intersect1d(ref_network.loc[ref_network['Gene1'] == g[0], 'Gene2'].values,
                                            ref_network.loc[ref_network['Gene2'] == g[1], 'Gene1'].values)
                if pair_trans.size > 0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate == True: gmasks[g] = ~ds.columns.isin(pair_trans)
                else: gmasks[g] = np.ones((ds.shape[1],)).astype(bool)

            # feedback loop or transitive (sequential) regulation
            elif self.args.motif == 'fbl-trans':
                pair_trans = np.intersect1d(ref_network.loc[ref_network['Gene1'] == g[1], 'Gene2'].values,
                                            ref_network.loc[ref_network['Gene2'] == g[0], 'Gene1'].values)
                if pair_trans.size > 0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate == True: gmasks[g] = ~ds.columns.isin(pair_trans)
                else: gmasks[g] = np.ones((ds.shape[1],)).astype(bool)

            # mutual interaction or simple interaction
            elif self.args.motif == 'mi-simple':
                pair_si = ref_network.loc[(ref_network['Gene1'] == g[1]) & (ref_network['Gene2'] == g[0]), :]
                if pair_si.shape[0] > 0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                gmasks[g] = np.ones((ds.shape[1],)).astype(bool)

        # generate list of tuples containing all possible TF-target gene pairs and highly-correlated neighbor genes (optional)
        gpairs = [tuple(list(g) + list(gpcorr.loc[g[0], (gpcorr.index.isin(tf)) & (~gpcorr.index.isin(g)) & (gmasks[g])].nlargest(self.args.neighbors).index)
                                + list(gpcorr.loc[g[1], (gpcorr.index.isin(tf)) & (~gpcorr.index.isin(g)) & (gmasks[g])].nlargest(self.args.neighbors).index))
                  for g in itertools.product(sorted(set(g1)), ds.columns)]
        random.seed(1234); random.shuffle(gpairs)

        # groups of TF-target gpairs for mini-batches
        if self.args.batch_size is not None:
            gpairs_batched = [gpairs[i : i + self.args.batch_size] for i in range(0, len(gpairs), self.args.batch_size)]
        else: gpairs_batched = [gpairs]

        # indices for generating pairwise joint-probability matrices
        matrix_gpairs = [[0,1], [0,0], [1,1]]
        for i in range(self.args.neighbors):
            matrix_gpairs.append([0, 2 + i])
            matrix_gpairs.append([1, 2 + self.args.neighbors + i])
        nchannels = len(matrix_gpairs) * (1 + self.args.max_lag)

        # loop over groups of gpairs to compile mini-batches: X, y, msk, g
        self.X_fn = [None] * len(gpairs_batched)
        self.y_fn = [None] * len(gpairs_batched)
        self.msk_fn = [None] * len(gpairs_batched)
        for j in tqdm(range(len(gpairs_batched))):
            X_fn_j = f'{self.outdir}/X_batch{j}_size{len(gpairs_batched[j])}.npy'
            y_fn_j = f'{self.outdir}/y_batch{j}_size{len(gpairs_batched[j])}.npy'
            msk_fn_j = f'{self.outdir}/msk_batch{j}_size{len(gpairs_batched[j])}.npy'
            g_fn_j = f'{self.outdir}/g_batch{j}_size{len(gpairs_batched[j])}.npy'

            # compile 4D array of TF-target gpair trajectories with optional neighbor genes
            gpairs_list_j = list(itertools.chain(*gpairs_batched[j]))
            if self.args.batch_size is None or j == len(gpairs_batched)-1: nsplit = len(gpairs_batched[j])
            else: nsplit = self.args.batch_size
            gpairs_ds_list = np.array_split(ds.loc[cell_idx, gpairs_list_j].values, nsplit, axis=1)
            gpairs_ds_list = [x.reshape(1, 2+2*self.args.neighbors, 1, cell_idx.size) for x in gpairs_ds_list]
            ds_batch_j = np.concatenate(gpairs_ds_list, axis=0).astype(np.float32)

            # compile gene names, regulation labels, and motif masks for gpairs
            g_batch_j = np.array([f'{g[0]} {g[1]}' for g in gpairs_batched[j]])
            y_batch_j = np.in1d(g_batch_j, ref_1d).reshape(ds_batch_j.shape[0], 1)
            msk_batch_j = np.in1d(g_batch_j, gpair_select).reshape(ds_batch_j.shape[0], 1)

            # compile 4D array containing stacks of 2D joint-probability matrices
            X_batch_j = np.zeros((ds_batch_j.shape[0], nchannels, self.args.nbins, self.args.nbins))
            for i in range(X_batch_j.shape[0]):
                for pair_idx in range(len(matrix_gpairs)):

                    # pseudotime-aligned joint-probability matrix
                    gpair = matrix_gpairs[pair_idx]
                    ds_gpair = np.squeeze(ds_batch_j[i, gpair, :, :]).T
                    if 0 in self.args.mask_lags: pass
                    else:
                        H, _ = np.histogramdd(ds_gpair, bins = (self.args.nbins, self.args.nbins))
                        H /= np.sqrt((H.flatten()**2).sum()) # L2-normalized matrix
                        X_batch_j[i, pair_idx * (1 + self.args.max_lag), :, :] = H

                    # pseudotime-lagged joint-probability matrices
                    for lag in range(1, self.args.max_lag + 1):
                        if lag in self.args.mask_lags: pass
                        else:
                            ds_gpair_lag = np.concatenate((ds_gpair[:-lag, 0].reshape(-1, 1),
                                                           ds_gpair[lag:, 1].reshape(-1, 1)), axis=1)
                            H, _ = np.histogramdd(ds_gpair_lag, bins = (self.args.nbins, self.args.nbins))
                            H /= np.sqrt((H.flatten()**2).sum()) # L2-normalized matrix
                            X_batch_j[i, pair_idx * (1 + self.args.max_lag) + lag, :, :] = H

            # mask specific regions of the joint-probability matrices [optional]
            if self.args.mask_region == 'off-off': 
                X_batch_j[:, :, :(self.args.nbins//2), :(self.args.nbins//2)] = 0.

            if self.args.mask_region in ['on-off', 'on']: 
                X_batch_j[:, :, (self.args.nbins//2):, :(self.args.nbins//2)] = 0.

            if self.args.mask_region in ['off-on', 'on']: 
                X_batch_j[:, :, :(self.args.nbins//2), (self.args.nbins//2):] = 0.

            if self.args.mask_region in ['on-on', 'on']: 
                X_batch_j[:, :, (self.args.nbins//2):, (self.args.nbins//2):] = 0.

            # save X, y, msk, and g as numpy files
            np.save(X_fn_j, X_batch_j.astype(np.float32))
            np.save(y_fn_j, y_batch_j.astype(np.float32))
            np.save(msk_fn_j, msk_batch_j.astype(np.float32))
            np.save(g_fn_j, g_batch_j.reshape(-1, 1))

            # save filenames
            self.X_fn[j] = X_fn_j
            self.y_fn[j] = y_fn_j
            self.msk_fn[j] = msk_fn_j