import glob, pathlib, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from pytorch_lightning.metrics.classification  import PrecisionRecallCurve as PRCurve
from pytorch_lightning.metrics.functional      import precision_recall_curve as prc
from pytorch_lightning.metrics.functional      import auc, roc

class Classifier(pl.LightningModule):
    """Deep neural network for binary classification of lagged gene-gene images"""
    def __init__(self, hparams, backbone, val_names, prefix):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.val_names = val_names
        self.prefix = prefix

        # store predictions & targets in pytortch_lightning precision-recall curve
        self.val_prc = nn.ModuleList([PRCurve(pos_label=1) for x in self.val_names])

    def list_dir(self, _dir_, _subdir_):
        return [str(x) for x in sorted(pathlib.Path(_dir_).glob(_subdir_))]

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
