import argparse
import torch
import numpy as np
import pandas as pd

import os
import shutil
import random
import itertools
import pickle
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    """Construct and/or load mini-batches of joint-probability matrices for the dataset"""

    def __init__(self, 
                 args: argparse.Namespace, 
                 root_dir: str,
                 split: str) -> None:

        self.args = args
        self.path = Path(root_dir)
        self.split = split

        # get mini-batch pathnames: X, y, msk (NEED TO TEST)
        if self.args.load_batches == True: ## need OR statement for second GPU to load batches
            _X_ = f'{self.split}/X_*.npy'
            _y_ = f'{self.split}/y_*.npy'
            _msk_ = f'{self.split}/msk_*.npy'
            self.X_fn = [str(x) for x in sorted(self.path.glob(_X_))]
            self.y_fn = [str(x) for x in sorted(self.path.glob(_y_))]
            self.msk_fn = [str(x) for x in sorted(self.path.glob(_msk_))]
        
        # construct mini-batches: X, y, mask
        else: self.construct_batches()

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

    def shuffle_pseudotime(self, pt: np.ndarray) -> np.ndarray:
        """Probabilistically swap cells' positions along trajectory at the given scale"""
        for i in np.arange(pt.size):
            j = np.random.normal(loc = 0, scale = self.args.shuffle * pt.size)
            _i_ = int(round(np.clip(i + j, 0, pt.size - 1)))
            pt[[i, _i_]] = pt[[_i_, i]]
        return pt

    def max_cross_correlation(self, a, v):
        corr = abs(np.correlate(a, v, 'same'))
        return corr[ corr.size//2 - self.args.max_lag : corr.size//2 ].max()

    def construct_batches(self) -> None:
        """Use sce to construct mini-batches for dataset and save as .npy files"""

        # load normalized data and PseudoTime files from sce dataset
        sce_pth = str(self.path)
        ds = pd.read_csv(f'{sce_pth}/NormalizedData.csv', index_col = 0).T
        ds.columns = ds.columns.str.lower() # gene names in lowercase
        pt = pd.read_csv(f"{sce_pth}/PseudoTime.csv", index_col = 0)
        out_pth = f'{sce_pth}/{self.split}'
        print(f'Constructing batches for {"/".join(out_pth.split("/")[-2:])}...')

        # load gene-regulation pairs from refNetwork file
        ref_network = pd.read_csv(f'{sce_pth}/refNetwork.csv')
        g1 = list(ref_network['Gene1'].str.lower().values)
        g2 = list(ref_network['Gene2'].str.lower().values)
        ref_1d = np.array([f'{g[0]} {g[1]}' for g in list(zip(g1, g2))])

        # list TFs from refNetwork or TranscriptionFactor file
        if self.args.predict == True:
            tf = np.loadtxt(f'{sce_pth}/TranscriptionFactors.csv', delimiter=',', dtype=str)
            tf = list(np.char.lower(tf))
        else: tf = g1.copy()

        # use TFs as Gene1 for gene pairs (START HERE - Gene1 selection)
        if self.args.predict == True and self.args.finetune == False: 
            g1 = tf.copy()

        ## labels for cross-validation or training/validation splits
            #try: labels = pd.read_csv(f'{item}/splitLabels.csv', index_col=0)
            #except: labels = None




        # remove previous results and create output directory
        if os.path.isdir(out_pth): shutil.rmtree(out_pth)
        os.mkdir(out_pth)

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




        ## start here

        # trajectory pairwise gene correlations (max absolute cross corr or max pearson corr)
        if self.args.max_lag > 0: traj_pcorr = sce.loc[cell_idx,:].corr(self.max_cross_correlation)
        elif self.args.max_lag==0: traj_pcorr = sce.loc[cell_idx,:].corr(method='pearson')

        # select gpairs in motif, optionally ablate (i.e. mask) genes
        gmasks, gpair_select = dict(), np.array([''])
        for g in itertools.product(sorted(set(g1)), sce.columns):
            if self.args.motif=='none':
                gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
            elif self.args.motif=='ffl-reg':
                pair_reg = ref.loc[ref['Gene2'].str.lower().isin(g), 'Gene1']
                pair_reg = pair_reg[ pair_reg.duplicated() ].str.lower().values
                if pair_reg.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate==True: gmasks[g] = ~sce.columns.isin(pair_reg)
                elif self.args.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
            elif self.args.motif=='ffl-tgt':
                pair_tgt = ref.loc[ref['Gene1'].str.lower().isin(g), 'Gene2']
                pair_tgt = pair_tgt[ pair_tgt.duplicated() ].str.lower().values
                if pair_tgt.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate==True: gmasks[g] = ~sce.columns.isin(pair_tgt)
                elif self.args.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
            elif self.args.motif=='ffl-trans':
                pair_trans = np.intersect1d(ref.loc[ref['Gene1'].str.lower()==g[0], 'Gene2'].str.lower().values,
                                            ref.loc[ref['Gene2'].str.lower()==g[1], 'Gene1'].str.lower().values)
                if pair_trans.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate==True: gmasks[g] = ~sce.columns.isin(pair_trans)
                elif self.args.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
            elif self.args.motif=='fbl-trans':
                pair_trans = np.intersect1d(ref.loc[ref['Gene1'].str.lower()==g[1], 'Gene2'].str.lower().values,
                                                ref.loc[ref['Gene2'].str.lower()==g[0], 'Gene1'].str.lower().values)
                if pair_trans.size>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                if self.args.ablate==True: gmasks[g] = ~sce.columns.isin(pair_trans)
                elif self.args.ablate==False: gmasks[g] = np.ones((sce.shape[1],)).astype(bool)
            elif self.args.motif=='mi-simple':
                pair_si = ref.loc[(ref['Gene1'].str.lower()==g[1])&(ref['Gene2'].str.lower()==g[0]),:]
                if pair_si.shape[0]>0: gpair_select = np.append(gpair_select, f'{g[0]} {g[1]}')
                gmasks[g] = np.ones((sce.shape[1],)).astype(bool)

        # generate list of tuples with all possible TF/target gpairs (opt, w/ neighbors)
        gpairs = [tuple( list(g) + list(traj_pcorr.loc[g[0],(traj_pcorr.index.isin(tf)) &
                                                            (~traj_pcorr.index.isin(g)) &
                                                            (gmasks[g])
                                                        ].nlargest(self.args.neighbors).index)
                                    + list(traj_pcorr.loc[g[1],(traj_pcorr.index.isin(tf)) &
                                                            (~traj_pcorr.index.isin(g)) &
                                                            (gmasks[g])
                                                        ].nlargest(self.args.neighbors).index) )
                    for g in itertools.product(sorted(set(g1)), sce.columns)]

        # shuffle order of TF/target gpairs
        seed = self.seed_from_string(traj_folder)
        random.seed(seed); random.shuffle(gpairs)

        # indices for pairwise comparisons
        gene_traj_pairs = [[0,1],[0,0],[1,1]]
        for i in range(self.args.neighbors):
            gene_traj_pairs.append([0,2+i])
            gene_traj_pairs.append([1,2+self.args.neighbors+i])

        # number of TF/target gpairs, no. cells
        n, n_cells = len(gpairs), cell_idx.size

        # generate TF/target gpair batches
        if self.args.batch_size is not None:
            gpairs_batched = [list(x) for x in self.grouper(gpairs, self.args.batch_size)]
            gpairs_batched = [list(filter(None, x)) for x in gpairs_batched]
        else: gpairs_batched = [gpairs]

        # loop over batches of TF/target gpairs
        for j in range(len(gpairs_batched)):
            X_fname = f'{traj_folder}/X_batch{j}_size{len(gpairs_batched[j])}.npy'
            y_fname = f'{traj_folder}/y_batch{j}_size{len(gpairs_batched[j])}.npy'
            msk_fname = f'{traj_folder}/msk_batch{j}_size{len(gpairs_batched[j])}.npy'
            g_fname = f'{traj_folder}/g_batch{j}_size{len(gpairs_batched[j])}.npy'

            # flatten TF/target gpairs (list of tuples) to list
            gpairs_list = list(itertools.chain(*gpairs_batched[j]))

            # split batch into single gpair examples (w/ neighbors)
            if self.args.batch_size is None or j==len(gpairs_batched)-1:
                sce_list = np.array_split(sce.loc[cell_idx, gpairs_list].values, len(gpairs_batched[j]), axis=1)
            else:
                sce_list = np.array_split(sce.loc[cell_idx, gpairs_list].values, self.args.batch_size, axis=1)

            # recombine re-shaped examples into full mini-batch
            sce_list = [g_sce.reshape(1,2+2*self.args.neighbors,1,n_cells) for g_sce in sce_list]
            X_batch = np.concatenate(sce_list, axis=0).astype(np.float32)

            # generate for batch: gene names, regulation labels, motif mask
            gpairs_batched_1d = np.array(['%s %s' % x[:2] for x in gpairs_batched[j]])
            y_batch = np.in1d(gpairs_batched_1d, ref_1d).reshape(X_batch.shape[0],1)
            msk_batch = np.in1d(gpairs_batched_1d, gpair_select).reshape(X_batch.shape[0],1)

            # generate 2D gene-gene co-expression images
            nchannels = len(gene_traj_pairs) * (1+self.args.max_lag)
            X_imgs = np.zeros((X_batch.shape[0], nchannels, self.args.nbins, self.args.nbins))

            # loop over examples in batch
            for i in range(X_imgs.shape[0]):

                # loop over gene-gene pairwise comparisons
                for pair_idx in range(len(gene_traj_pairs)):

                    # aligned gene-gene co-expression image
                    pair = gene_traj_pairs[pair_idx]
                    data = np.squeeze(X_batch[i,pair,:,:]).T
                    if 0 in self.args.mask_lags: pass
                    else:
                        H, _ = np.histogramdd(data, bins=(self.args.nbins, self.args.nbins))
                        H /= np.sqrt((H.flatten()**2).sum())
                        X_imgs[i,pair_idx*(1+self.args.max_lag),:,:] = H

                    # lagged gene-gene co-expression images
                    for lag in range(1,self.args.max_lag+1):
                        if lag in self.arg.mask_lags: pass
                        else:
                            data_lagged = np.concatenate((data[:-lag,0].reshape(-1,1),
                                                            data[lag:,1].reshape(-1,1)), axis=1)
                            H, _ = np.histogramdd(data_lagged, bins=(self.args.nbins, self.args.nbins))
                            H /= np.sqrt((H.flatten()**2).sum())
                            X_imgs[i,pair_idx*(1+self.args.max_lag)+lag,:,:] = H

            # optionally, mask region(s)
            if self.args.mask_region=='off-off':
                X_imgs[:,:,:(self.args.nbins//2),:(self.args.nbins//2)] = 0.
            if self.args.mask_region in ['on-off', 'on']:
                X_imgs[:,:,(self.args.nbins//2):,:(self.args.nbins//2)] = 0.
            if self.args.mask_region in ['off-on', 'on']:
                X_imgs[:,:,:(self.args.nbins//2),(self.args.nbins//2):] = 0.
            if self.args.mask_region in ['on-on', 'on']:
                X_imgs[:,:,(self.args.nbins//2):,(self.args.nbins//2):] = 0.
            if self.args.mask_region == 'edges':
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