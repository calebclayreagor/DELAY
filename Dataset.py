import warnings
import random
import itertools
import torch
import numba
import numpy as np
import pandas as pd
import anndata as ad

from typing             import Tuple
from typing             import TypeVar
from typing             import Optional
from types              import SimpleNamespace
from tqdm               import tqdm
from pynndescent        import NNDescent
from scipy.sparse       import issparse

Self = TypeVar('Self', bound = 'Dataset')

@numba.njit
def max_cross_corr_dist(a : np.array,
                        b : np.array,
                        eps: float = 1e-10
                        ) -> float:
    """
    Compute the distance between two pseudotime-lagged gene-expression
    trajectories using the maximum cross-correlation.
    """
    return -np.log(np.abs(np.correlate(a[:-5], b)).max() + eps)

class Dataset(torch.utils.data.Dataset):
    """
    Internal dataset class for DELAY.

    Compiles mini-batches of pseudotime-lagged joint-probability matrices
    from an AnnData object, optionally using ground-truth TF-target pairs.

    Note:
        Memory usage scales with the product of the number of transcription factors
        and target genes. For large datasets, this may result in high memory demand
        during mini-batch compilation.
    
    This class is used internally by DELAY_Classifier and is not intended
    for direct use by end users.
    """

    def __init__(self : Self,
                 adata : ad.AnnData,
                 args : SimpleNamespace,
                 name : Optional[str] = None,
                 y : Optional[pd.DataFrame] = None
                 ) -> Self:
        
        # pseudotime values (sorted)
        t = adata.obs[args.t].copy()
        t_sorted_ix = t.sort_values().index

        # TF, target genes
        tf = adata.var_names[adata.var[args.mask_tf]]
        target = adata.var_names[adata.var[args.mask_target]]
        genes = np.union1d(tf, target)

        # adata (t-sorted; TF/target only), X (full & TF-only)
        adata = adata[t_sorted_ix, genes].copy()
        if args.layer is None: X = adata.X.copy()
        else: X = adata.layers[args.layer].X.copy()
        if issparse(X):
            warnings.warn("Input expression matrix is sparse and will be densified. This may increase memory usage substantially.", RuntimeWarning)
            X = X.toarray()
        X_tf_only = X[:, adata.var_names.get_indexer(tf)].copy()

        # TFs for gene pairs
        if y is not None:

            # training and validation
            tf_gpair = np.asarray(sorted(y.TF.unique()))
            y_arr = y.agg(' '.join, axis = 1).values  # 1d reference array for positive pairs

        else:
            # prediction
            tf_gpair = np.asarray(sorted(tf.copy()))

        n_gpairs = (tf_gpair.size * target.size)
        if n_gpairs > 500000:
            warnings.warn(f"Compiling features for {n_gpairs} TF-target pairs. This may lead to high memory usage. Consider using fewer TFs/targets.", RuntimeWarning)

        # approximate search for neighbor genes (TFs)
        nn_model = NNDescent(X_tf_only.T, metric = max_cross_corr_dist, random_state = 0)
        nn_ix, _ = nn_model.query(X.T, k = 5)

        # list of tuples for TF-target gene pairs and neighbors
        gpairs = [None] * n_gpairs
        for i in range(tf_gpair.size):
            mask_tf_i = np.isin(genes, [tf_gpair[i]])
            tf_i_nbrs = tf[nn_ix[mask_tf_i].flatten()]
            for j in range(target.size):
                gpair_i_j = [tf_gpair[i], target[j]]
                mask_target_j = np.isin(genes, [target[j]])
                target_j_nbrs = tf[nn_ix[mask_target_j].flatten()]
                tf_i_nbrs_gpair = tf_i_nbrs[~np.isin(tf_i_nbrs, gpair_i_j)][:2].tolist()
                target_j_nbrs_gpair = target_j_nbrs[~np.isin(target_j_nbrs, gpair_i_j)][:2].tolist()
                gpairs[(i * target.size) + j] = tuple(gpair_i_j + tf_i_nbrs_gpair + target_j_nbrs_gpair)
        random.shuffle(gpairs)
        gpairs_batched = [gpairs[i : i + args.batch_size] for i in range(0, len(gpairs), args.batch_size)]

        # compile mini-batches: X(2d) — features, y — targets, g — gene names
        self.X, self.y, self.g = [[None] * len(gpairs_batched) for _ in range(3)]
        matrix_gpairs = [[0, 1], [0, 0], [1, 1], [0, 2], [1, 4], [0, 3], [1, 5]]
        for j in tqdm(range(len(gpairs_batched)), desc = name):

            # gene-expression trajectories with neighbor genes
            gpairs_list_j = list(itertools.chain(*gpairs_batched[j]))
            nsplit = len(gpairs_batched[j])
            X_gpairs = np.array_split(X[:, adata.var_names.get_indexer(gpairs_list_j)], nsplit, axis = 1)
            X_gpairs = [arr.reshape(1, 6, 1, X.shape[0]) for arr in X_gpairs]
            X_j = np.concatenate(X_gpairs, axis = 0).astype(np.float32)

            # gene names (g)
            g_j = np.array([g[:2] for g in gpairs_batched[j]])
            g_j_arr = np.array([f'{g[0]} {g[1]}' for g in gpairs_batched[j]])

            # class labels (y)
            if y is not None:
                y_j = np.in1d(g_j_arr, y_arr).reshape(X_j.shape[0], 1)

            # stack of joint-probability matrices (X2d)
            #   batch_size x nchannels x nbins_histogram x nbins_histogram
            X2d_j = np.zeros((X_j.shape[0], 42, args.nbins_histogram, args.nbins_histogram))
            for i in range(X_j.shape[0]):
                for pair_idx in range(len(matrix_gpairs)):

                    # no pseudotime lag
                    gpair = matrix_gpairs[pair_idx]
                    X_gpair_idx = np.squeeze(X_j[i, gpair, :, :]).T
                    H, _ = np.histogramdd(X_gpair_idx, bins = (args.nbins_histogram, args.nbins_histogram))
                    H /= np.sqrt((H.flatten() ** 2).sum()) # L2-normalized matrix
                    X2d_j[i, pair_idx * 6, :, :] = H

                    # pseudotime lagged
                    for lag in range(1, 6):
                        X_gpair_lag = np.concatenate((X_gpair_idx[ : -lag, 0].reshape(-1, 1),
                                                      X_gpair_idx[lag : , 1].reshape(-1, 1)), axis = 1)
                        H, _ = np.histogramdd(X_gpair_lag, bins = (args.nbins_histogram, args.nbins_histogram))
                        H /= np.sqrt((H.flatten() ** 2).sum()) # L2-normalized matrix
                        X2d_j[i, pair_idx * 6 + lag, :, :] = H

            # save mini-batch
            self.X[j] = X2d_j.astype(np.float32)
            if y is not None:
                self.y[j] = y_j.astype(np.float32)
            self.g[j] = g_j

    def __len__(self : Self) -> int:
        """Return number of mini-batches."""
        return len(self.X)

    def __getitem__(self : Self, ix : int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return features X, targets y, and gene-pair names for the given batch index."""
        return self.X[ix], self.y[ix], self.g[ix]