import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import lightning as L

from Dataset                        import Dataset
from Classifier                     import Classifier
from vgg                            import VGG
from typing                         import TypeVar
from typing                         import Iterable
from typing                         import Union
from typing                         import Optional
from types                          import SimpleNamespace
from torch.utils.data               import DataLoader
from torch.utils.data               import ConcatDataset
from lightning.pytorch.loggers      import CSVLogger
from lightning.pytorch.callbacks    import ModelCheckpoint
from lightning.pytorch.callbacks    import EarlyStopping

Self = TypeVar('Self', bound = 'DELAY_Classifier')

class DELAY_Classifier:

    def __init__(self : Self,
                 random_state : int = 1234,
                 ) -> Self:
        L.seed_everything(random_state)
        self.datasets = dict()
        self.datasets_log = dict()
        self.trained_models = dict()
        self.training_log = dict()

    def compile_batches(self : Self,
                        adata : ad.AnnData,
                        *,
                        layer : Optional[str] = None,
                        t : str = 't',
                        mask_tf : str = 'DELAY_TF',
                        mask_target : str = 'DELAY_Target',
                        dataset_name : Optional[str] = None,
                        y : Optional[pd.DataFrame] = None,
                        val_var : Optional[Union[str, Iterable[str]]] = None,
                        batch_size : int = 32,
                        nbins_histogram : int = 32
                        ) -> None:
        """
        Compiles mini-batches for training and validation or prediction datasets.

        Performs input checks, logs metadata, and constructs >=1 Dataset object
        to be stored in `self.datasets`. If `y` is provided, data is split into
        training and validation sets based on `val_var`. If not, a prediction-only
        dataset is compiled.

        Args:
            adata (anndata.AnnData):
                Annotated single-cell gene expression dataset. Must contain:
                - `.obs[t]`: a float-valued pseudotime or time column
                - `.var[mask_tf]`: a boolean column indicating TFs and cofactors
                - `.var[mask_target]`: a boolean column indicating target genes

            layer (str, optional):
                Name of the layer in `adata.layers` to use for expression. If None, uses `adata.X`.

            t (str, default = 't'):
                Key in `adata.obs` for the pseudotime or time values.

            mask_tf (str, default = 'DELAY_TF'):
                Column in `adata.var` indicating transcription factors. Must be boolean.

            mask_target (str, default = 'DELAY_Target'):
                Column in `adata.var` indicating target genes. Must be boolean.

            dataset_name (str, optional):
                Identifier for the dataset. If None, a name is auto-generated.

            y (pandas.DataFrame, optional):
                DataFrame of known TF-target regulatory pairs, with columns:
                - 'TF': transcription factor gene names
                - 'Target': target gene names
                All TFs must be present in `adata.var_names[mask_tf]` and all targets must be present in
                `adata.var_names[mask_target]`.

            val_var (str or Iterable[str], optional):
                TFs to hold out for validation. Required if `y` is provided. Must be a subset of `y['TF']`.

            batch_size (int, default = 32):
                Number of TF-target gene pairs per mini-batch. Must be a positive integer.

            nbins_histogram (int, default = 32):
                Number of bins used to generate co-expression matrices. Must be >= 16.

        Raises:
            TypeError: If inputs have incorrect types.
            ValueError: If required keys or conditions are not met.
            Warning: If any TFs regulate fewer than 10% of the target gene set.

        Side Effects:
            - Updates `self.datasets` with one or more compiled Dataset objects.
            - Updates `self.datasets_log[dataset_name]` with arguments and additional information.

        Notes:
            - This function will densify any sparse expression matrices (e.g., `adata.X` or `adata.layers[layer]`).
              This may lead to high memory usage for large datasets. Use with caution if input is sparse.
            - Memory usage also scales with the product of the number of transcription factors and target genes.
              If either set is very large, the number of TF-target pairs can become substantial, which may cause
              excessive memory use during feature compilation. Consider filtering to a smaller set of TFs or targets
              if running into resource constraints.
        """

        ## To-Do: mask_obs (variable, checks, implementation, documentation)
        
        # check dataset name
        if dataset_name is None:
            dataset_name = 'Dataset' + str(len(self.datasets) + 1)
        if dataset_name in self.datasets_log:
            raise ValueError(f"Dataset name '{dataset_name}' already exists.")
        self.datasets_log[dataset_name] = dict()
        
        # check `adata` type
        if not isinstance(adata, ad.AnnData):
            raise TypeError(f"`adata` must be an AnnData object, got type {type(adata)}.")

        # check `layer` exists
        if (layer is not None) and (layer not in adata.layers):
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")

        # check `t` exists, dtype
        if t not in adata.obs:
            raise ValueError(f"Column '{t}' not found in adata.obs.")
        if not pd.api.types.is_float_dtype(adata.obs[t]):
            raise ValueError(f"`adata.obs['{t}']` does not contain float dtype.")

        # check TF/target masks exist, dtype
        for key in (mask_tf, mask_target):
            if key not in adata.var:
                raise ValueError(f"Column '{key}' not found in adata.var.")
            if not pd.api.types.is_bool_dtype(adata.var[key]):
                raise ValueError(f"`adata.var['{key}']` does not contain boolean dtype.")
            n_gene_key = adata.var[key].sum()
            if n_gene_key == 0:
                raise ValueError(f"`adata.var['{key}']` contains no `True` entries (>=1 required).")
            self.datasets_log[dataset_name][f'n_{key}'] = n_gene_key

        # check ground truth
        if y is not None:
            
            # check `y` type, TF/target columns exist, contain valid genes
            if not isinstance(y, pd.DataFrame):
                raise TypeError(f"`y` must be a DataFrame object, got type {type(y)}.")
            for key in ('TF', 'Target'):
                if key not in y.columns:
                    raise ValueError(f"`y` does not contain '{key}' column.")
            bad_tf = sorted(set(y.TF) - set(adata.var_names[adata.var[mask_tf]]))
            if len(bad_tf) > 0:
                raise ValueError(f"The following TFs from `y` were not found in 'mask_tf': {bad_tf}.")
            bad_target = sorted(set(y.Target) - set(adata.var_names[adata.var[mask_target]]))
            if len(bad_target) > 0:
                raise ValueError(f"The following Targets from `y` were not found in 'mask_target': {bad_target}.")

            # check `val_var` exists, type, contains valid genes
            if val_var is None:
                raise ValueError("`val_var` must be provided when `y` is not None.")
            val_var = [val_var] if isinstance(val_var, str) else val_var
            bad_val_type = [var for var in val_var if not isinstance(var, str)]
            if len(bad_val_type) > 0:
                raise TypeError(f"`val_var` contains the following non-string entries: {bad_val_type}.")
            val_var = sorted(set(val_var))
            bad_val = sorted(set(val_var) - set(y.TF))
            if len(bad_val) > 0:
                raise ValueError(f"The following TFs from `val_var` were not found `y`: {bad_val}.")
            if len(val_var) == 0:
                raise ValueError("`val_var` must contain at least one valid entry.")
            self.datasets_log[dataset_name]['n_val_var'] = len(val_var)

            # check training TFs exist
            train_var = sorted(set(y.TF) - set(val_var))
            if len(train_var) <= 0:
                raise ValueError("`y` must contain >=1 TF for training.")
            self.datasets_log[dataset_name]['train_var'] = train_var
            self.datasets_log[dataset_name]['n_train_var'] = len(train_var)
            
            # check positive-class prevalence
            tf_pos_frac = y.groupby('TF').size() / adata.var[mask_target].sum()
            low_pos_frac = tf_pos_frac[tf_pos_frac < .1]
            if not low_pos_frac.empty:
                warnings.warn(f"The following TFs from `y` have <10% true interactions: {low_pos_frac.to_dict()}")
            self.datasets_log[dataset_name]['positive_class_prevalence'] = tf_pos_frac.to_dict()

        # check `batch_size` type, value
        if (not isinstance(batch_size, int)) or (batch_size < 1):
            raise ValueError(f"`batch_size` must be a positive integer, got {batch_size}.")

        # check `nbins_histogram` type, value
        if (not isinstance(nbins_histogram, int)) or (nbins_histogram < 16):
            raise ValueError(f"`nbins_histogram` must be an integer >=16, got {nbins_histogram}.")

        # save checked args
        args = SimpleNamespace(
            layer = layer,
            t = t,
            mask_tf = mask_tf,
            mask_target = mask_target,
            dataset_name = dataset_name,
            val_var = val_var,
            batch_size = batch_size,
            nbins_histogram = nbins_histogram)
        self.datasets_log[dataset_name].update(vars(args))

        # compile mini-batches
        if y is not None:

            # training
            train_key = dataset_name + '_training'
            self.datasets_log[dataset_name]['train_key'] = train_key
            y_train = y.loc[np.isin(y.TF, train_var)]
            self.datasets[train_key] = Dataset(adata = adata,
                                               args = args,
                                               name = train_key,
                                               y = y_train)
            
            # validation
            val_key = dataset_name + '_validation'
            self.datasets_log[dataset_name]['val_key'] = val_key
            y_val = y.loc[np.isin(y.TF, val_var)]
            self.datasets[val_key] = Dataset(adata = adata,
                                             args = args,
                                             name = val_key,
                                             y = y_val)
            
        else:
            # prediction
            self.datasets[dataset_name] = Dataset(adata = adata,
                                                  args = args,
                                                  name = dataset_name)
            
    def fit(self : Self,
            *,
            model_name : Optional[str] = None,
            training_datasets_names : Optional[Union[str, Iterable[str]]] = None,
            validation_datasets_names : Optional[Union[str, Iterable[str]]] = None,
            num_workers : Optional[int] = None,
            use_atac_model : bool = False,
            lr : float = 0.1,
            max_epochs : int = 200,
            accelerator : str = 'auto',
            devices : Union[str, int, list] = 'auto',
            trainer : Optional[L.Trainer] = None,
            **trainer_kwargs
            ) -> None:
        
        # check model name
        if model_name is None:
            model_name = 'Model' + str(len(self.trained_models) + 1)
        if model_name in self.training_log:
            raise ValueError(f"Model name '{model_name}' already exists.")
        self.training_log[model_name] = dict()

        # check training, validation dataset names
        for split in ('training', 'validation'):
            key = split + '_datasets_names'
            names = locals()[key]
            if isinstance(names, str):
                names = [names]
            elif names is None:
                names = [k for k in self.datasets if k.endswith(f'_{split}')]
            missing = sorted(set(names) - set(self.datasets))
            if len(missing) > 0:
                raise ValueError(f"Unable to find {split} dataset(s) in self.datasets: {missing}.")
            if split == 'training':
                training_datasets_names = names
            else:
                validation_datasets_names = names

        # check `num_workers` type, value
        if num_workers is None:
            num_workers = min(2, os.cpu_count() or 1)
        elif (not isinstance(num_workers, int)) or (num_workers < 1):
            raise TypeError(f"`num_workers` must be a positive integer, got {num_workers}.")

        # load training datasets
        training_datasets = [self.datasets[key] for key in training_datasets_names]
        training_datasets_merge = ConcatDataset(training_datasets)
        training_datasets_loader = DataLoader(training_datasets_merge,
                                              batch_size = None,
                                              shuffle = True,
                                              num_workers = num_workers,
                                              pin_memory = True,
                                              persistent_workers = True)
        
        # load validation datasets
        validation_datasets = [self.datasets[key] for key in validation_datasets_names]
        validation_datasets_loader = [DataLoader(ds,
                                                 batch_size = None,
                                                 num_workers = num_workers,
                                                 pin_memory = True,
                                                 persistent_workers = True)
                                      for ds in validation_datasets]
        
        # check `use_atac_model` type
        if not isinstance(use_atac_model, bool):
            raise TypeError(f"`use_atac` must be a boolean, got {type(use_atac_model)}.")
        
        # check `lr` type, value
        if (not isinstance(lr, float)) or (lr <= 0):
            raise ValueError(f"`lr` must be a positive float, got {lr}.")
        
        # check `max_epochs` type, value
        if (not isinstance(max_epochs, int)) or (max_epochs < 1):
            raise ValueError(f"`max_epochs` must be an integer >=1, got {max_epochs}.")

        # check `trainer` type
        if trainer is not None:
            if not isinstance(trainer, L.Trainer):
                raise TypeError(f"`trainer` must be a `lightning.Trainer` instance, got {type(trainer)}.")
            
        else:
            # check `accelerator` type
            if not isinstance(accelerator, str):
                raise TypeError(f"`accelerator` must be a string, got {type(accelerator)}.")
            accelerator = accelerator.lower()
                 
            callbacks = [ModelCheckpoint(dirpath = os.path.join('DELAY_Classifier', model_name),
                                       filename = 'best_model_epoch={epoch:02d}_val_F1={val_F1:.4f}',
                                       monitor = 'val_F1', mode = 'max', save_top_k = 1),
                         EarlyStopping(monitor = 'val_F1', mode = 'max', min_delta = .01, patience = 10)]
            
            trainer = L.Trainer(accelerator = accelerator, devices = devices, max_epochs = max_epochs, 
                         num_sanity_val_steps = 0, log_every_n_steps = loss_freq, deterministic = 'warn', 
                         callbacks = callbacks, logger = CSVLogger('RESULTS', name = args.outdir))

        
        # set up classifier
        net = VGG(cfg = [1024, 'M', 512, 'M', 256, 'M', 128, 'M', 64], in_channels = 42)
        if use_atac_model:
            model_fn = os.path.join('Models', 'Mannens2023-ATAC.ckpt')
        else:
            model_fn = os.path.join('Models', 'Reagor2023-RNA.ckpt')
        model = Classifier.load_from_checkpoint(args.model, hparams = args, backbone = net, valnames = valnames, prefix = prefix)

        trainer.fit(model, train_loader, val_loader)

         
# trainer.predict(model, train_loader)