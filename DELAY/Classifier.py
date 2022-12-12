import argparse
import pathlib
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Union
from typing import List
from typing import Tuple
from typing import TypeVar

from pytorch_lightning.metrics.classification  import PrecisionRecallCurve as PRCurve
from pytorch_lightning.metrics.functional      import precision_recall_curve as prc
from pytorch_lightning.metrics.functional      import auc, roc

from Networks.VGG_CNNC import VGG_CNNC
from Networks.SiameseVGG import SiameseVGG
from Networks.vgg import VGG

Self = TypeVar('Self', bound = 'Classifier')

## TO-DO: DOUBLE CHECK HPARAM NAMES/TYPES

class Classifier(pl.LightningModule):
    """Deep neural network for classification of TF-target joint-probability matrices"""

    def __init__(self: Self,
                 hparams: argparse.Namespace, 
                 backbone: Union[VGG, SiameseVGG, VGG_CNNC], 
                 valnames: List[str],
                 prefix: str) -> Self:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.valnames = valnames
        self.prefix = prefix

        # store predictions/targets in pytortch_lightning precision-recall curve
        self.val_prc = nn.ModuleList([PRCurve(pos_label = 1) for _ in self.valnames])

    def configure_optimizers(self: Self) -> torch.optim.sgd.SGD:
        return torch.optim.SGD(self.parameters(), lr = self.hparams.learning_rate)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self: Self,
                      train_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]],
                      batch_idx: int
                      ) -> torch.Tensor:
        X, y, _, _ = train_batch
        out = self.forward(X)
        loss = F.binary_cross_entropy_with_logits(out, y, weight = y.sum()/y.size(0), reduction = 'sum')/self.hparams.batch_size
        self.log('train_loss', loss, on_step = True, on_epoch = True, sync_dist = True, add_dataloader_idx = False)
        return loss

    def validation_step(self: Self,
                        val_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]],
                        batch_idx: int,
                        dataset_idx: int = 0
                        ) -> None:
        X, y, _, _ = val_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)
        loss = F.binary_cross_entropy_with_logits(out, y, weight = y.sum()/y.size(0), reduction = 'sum')/self.hparams.batch_size
        self.log(f'{self.prefix}loss', loss, on_step = False, on_epoch = True, sync_dist = True, add_dataloader_idx = False)

        # update precision-recall curve
        self.val_prc[dataset_idx](pred, y)

    def test_step(self: Self,
                  test_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]],
                  batch_idx: int,
                  dataset_idx: int = 0
                  ) -> None:
        X, y, msk, fn = test_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)

        # update precision-recall curve for unmasked gpairs (optional)
        if msk.sum() > 0:
            pred_msk = torch.masked_select(pred, msk > 0)
            y_msk = torch.masked_select(y, msk > 0)
            self.val_prc[dataset_idx](pred_msk, y_msk)

        # save predictions for testing mini-batch as .npy file
        fn_out = f'{fn[0]}pred_k={self.hparams.valsplit}_{fn[1]}'
        np.save(fn_out, pred.cpu().detach().numpy().astype(np.float32), allow_pickle = True)

    def predict_step(self: Self,
                     pred_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]],
                     batch_idx: int,
                     dataset_idx: int = 0
                     ) -> None:
        X, _, _, fn = pred_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)

        # save predictions for mini-batch as .npy file
        fn_out = f'{fn[0]}pred_{fn[1]}'
        np.save(fn_out, pred.cpu().detach().numpy().astype(np.float32), allow_pickle = True)





    ## start here
    
    def list_dir(self, _dir_, _subdir_):
        return [str(x) for x in sorted(pathlib.Path(_dir_).glob(_subdir_))]

    def on_validation_epoch_end(self):
        val_auprc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        val_auroc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        for idx in range(len(self.valnames)):
            name = self.valnames[idx]
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
        test_auprc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        test_auroc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        for idx in range(len(self.valnames)):
            name = self.valnames[idx]
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

    def on_predict_end(self):
        save_dir = f'lightning_logs/{self.hparams.output_dir}/'
        if self.hparams.data_type=='scrna-seq':
            sce_fname = self.list_dir(self.hparams.datasets_dir,
                            f'*/*/*/*/ExpressionData.csv')
        elif self.hparams.data_type=='scatac-seq':
            sce_fname = self.list_dir(self.hparams.datasets_dir,
                            f'*/*/*/*/AccessibilityData.csv')
        tfs_fname = self.list_dir(self.hparams.datasets_dir,
                        f'*/*/*/*/TranscriptionFactors.csv')
        genes = pd.read_csv(sce_fname[0], index_col=0).index.str.lower()
        tfs = pd.read_csv(tfs_fname[0], index_col=0).index.str.lower()
        probs_matrix = pd.DataFrame(0., index=tfs, columns=genes)
        pred_fnames = self.list_dir(self.hparams.datasets_dir,
            f'*/*/*/*/*/pred_seed={self.hparams.global_seed}*.npy')
        for idx in range(len(pred_fnames)):
            path = pred_fnames[idx].split('/')
            g_fname = '/'.join(path[0:-1]) + '/g_' + '_'.join(path[-1].split('_')[2:])
            pred = np.load(pred_fnames[idx], allow_pickle=True).reshape(-1)
            g = np.load(g_fname, allow_pickle=True).reshape(-1)
            g = np.stack(np.char.split(g, ' '), axis=0)
            ii = np.where(g[:,0][:,None]==tfs[None,:])[1]
            jj = np.where(g[:,1][:,None]==genes[None,:])[1]
            probs_matrix.values[ii, jj] = pred
        probs_matrix.to_csv(
            f'{save_dir}predicted_probabilities_seed={self.hparams.global_seed}.csv')