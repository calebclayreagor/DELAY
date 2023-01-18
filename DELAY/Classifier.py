import argparse
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pathlib import Path
from typing import Union
from typing import List
from typing import Tuple
from typing import TypeVar

from torchmetrics import AveragePrecision, AUROC
from Networks.VGG_CNNC import VGG_CNNC
from Networks.SiameseVGG import SiameseVGG
from Networks.vgg import VGG

Self = TypeVar('Self', bound = 'Classifier')

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

        # set up separate torch.nn ModuleLists to update/compute AUC values for PR/ROC metrics
        self.val_auprc = nn.ModuleList([AveragePrecision(task = 'binary') for _ in self.valnames])
        self.val_auroc = nn.ModuleList([AUROC(task = 'binary') for _ in self.valnames])

    def configure_optimizers(self: Self) -> torch.optim.SGD:
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
        self.log('train_loss', loss, on_step = True, on_epoch = True, batch_size = X.size(0), sync_dist = True, add_dataloader_idx = False)
        return loss

    def validation_step(self: Self,
                        val_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]],
                        batch_idx: int,
                        dataset_idx: int = 0
                        ) -> None:
        X, y, _, _ = val_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)

        # update torch.nn modules for AUC metrics
        self.val_auprc[dataset_idx].update(pred, y)
        self.val_auroc[dataset_idx].update(pred, y)

    def test_step(self: Self,
                  test_batch: Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]],
                  batch_idx: int,
                  dataset_idx: int = 0
                  ) -> None:
        X, y, msk, fn = test_batch
        out = self.forward(X)
        pred = torch.sigmoid(out)

        # update modules for AUC metrics across unmasked gpairs (optional)
        if msk.sum() > 0:
            pred_msk = torch.masked_select(pred, msk > 0)
            y_msk = torch.masked_select(y, msk > 0)
            self.val_auprc[dataset_idx].update(pred_msk, y_msk)
            self.val_auroc[dataset_idx].update(pred_msk, y_msk)

        # save predictions for mini-batch as .npy file
        fn_out = f'{fn[0]}pred_k={self.hparams.valsplit}_{fn[1]}'
        np.save(fn_out, pred.cpu().detach().numpy().astype(np.float32))

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
        np.save(fn_out, pred.cpu().detach().numpy().astype(np.float32))

    def on_validation_epoch_end(self: Self) -> None:
        """Compute dataset and average AUC values from updated torch.nn ModuleLists for PR/ROC metrics"""
        val_auprc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        val_auroc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        for idx in range(len(self.valnames)):
            val_auprc[idx], val_auroc[idx] = self.val_auprc[idx].compute(), self.val_auroc[idx].compute()
            self.log(f'{self.valnames[idx]}auprc', val_auprc[idx], sync_dist = True, add_dataloader_idx = False)
            self.log(f'{self.valnames[idx]}auroc', val_auroc[idx], sync_dist = True, add_dataloader_idx = False)
            self.val_auprc[idx].reset(); self.val_auroc[idx].reset() # reset metric modules for next epoch
        avg_auprc, avg_auroc = val_auprc.mean(), val_auroc.mean()
        self.log(f'{self.prefix}avg_auprc', avg_auprc, sync_dist = True, add_dataloader_idx = False)
        self.log(f'{self.prefix}avg_auroc', avg_auroc, sync_dist = True, add_dataloader_idx = False)
        self.log(f'{self.prefix}avg_auc', (avg_auprc + avg_auroc)/2, sync_dist = True, add_dataloader_idx = False)

    def on_test_epoch_end(self: Self) -> None:
        """Compute dataset and average AUC values from updated torch.nn ModuleLists for PR/ROC metrics"""
        test_auprc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        test_auroc = torch.zeros((len(self.valnames),), device = torch.cuda.current_device())
        for idx in range(len(self.valnames)):
            target_idx = self.val_auprc[idx].target
            input(target_idx)
            test_auprc[idx], test_auroc[idx] = self.val_auprc[idx].compute(), self.val_auroc[idx].compute()
            self.log(f'_{self.valnames[idx]}auprc', test_auprc[idx], sync_dist = True, add_dataloader_idx = False)
            self.log(f'_{self.valnames[idx]}auroc', test_auroc[idx], sync_dist = True, add_dataloader_idx = False)
            self.log(f'_{self.valnames[idx]}_density', target_idx.sum()/target_idx.size(0), sync_dist = True, add_dataloader_idx = False)
            self.val_auprc[idx].reset(); self.val_auroc[idx].reset()
        avg_auprc, avg_auroc = test_auprc.mean(), test_auroc.mean()
        self.log(f'_{self.prefix}avg_auprc', avg_auprc, sync_dist = True, add_dataloader_idx = False)
        self.log(f'_{self.prefix}avg_auroc', avg_auroc, sync_dist = True, add_dataloader_idx = False)
        self.log(f'_{self.prefix}avg_auc', (avg_auprc + avg_auroc)/2, sync_dist = True, add_dataloader_idx = False)

    def on_predict_end(self: Self) -> None:
        """Compile and save final matrix of gene-regulatory predictions from mini-batches"""
        ds_dir = f'{self.hparams.datadir}'
        tf_fn = glob.glob(f'{ds_dir}*/TranscriptionFactors.csv')[0]
        ds_fn = glob.glob(f'{ds_dir}*/NormalizedData.csv')[0]
        g1 = np.char.lower(np.loadtxt(tf_fn, delimiter = ',', dtype = str))
        g2 = pd.read_csv(ds_fn, index_col = 0).index.str.lower()
        pred_mat = pd.DataFrame(0., index = g1, columns = g2)
        pred_fn = list(map(str, sorted(Path(ds_dir).glob(f'*/prediction/pred_*.npy'))))
        g_fn = list(map(str, sorted(Path(ds_dir).glob(f'*/prediction/g_*.npy'))))
        for j in range(len(pred_fn)):
            pred_j = np.load(pred_fn[j]).reshape(-1)
            g_j = np.load(g_fn[j]).reshape(-1)
            g_j = np.stack(np.char.split(g_j, ' '), axis = 0)
            ii = np.where(g_j[:, 0][:, None] == g1[None, :])[1]
            jj = np.where(g_j[:, 1][:, None] == g2[None, :])[1]
            pred_mat.values[ii, jj] = pred_j
        pred_mat.to_csv(f'RESULTS/{self.hparams.outdir}/regPredictions.csv')