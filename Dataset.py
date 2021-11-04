import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import os, sys
import glob
import pathlib
import shutil
import random
import itertools
import pickle

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
