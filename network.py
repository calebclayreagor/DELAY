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
from pytorch_lightning.metrics.functional      import auc, precision_recall_curve as prc


class Dataset(torch.utils.data.Dataset):
    """Dataset class for generating/loading batches"""
    def __init__(self, root_dir, rel_path, batchSize=None, wShuffle=0.02,
                        minCells=40, overwrite=False, load_prev=True):

        self.root_dir = root_dir
        self.rel_path = rel_path
        self.batch_size = batchSize
        self.overwrite = overwrite
        self.load_prev = load_prev
        self.pt_shuffle = wShuffle
        self.min_cells = minCells

        self.sce_fnames = sorted(pathlib.Path(self.root_dir).glob(self.rel_path))

        if self.load_prev==True:
            prev_path = '/'.join(self.rel_path.split('/')[:-1])+'*/'
            self.X_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'X_*.npy'))]
            self.y_fnames = [str(x) for x in sorted(pathlib.Path(self.root_dir).glob(prev_path+'y_*.npy'))]

        else:
            self.X_fnames = [None] * len(self.sce_fnames)
            self.y_fnames = [None] * len(self.sce_fnames)
            for sce_fname in tqdm(self.sce_fnames):
                self.generate_batches(str(sce_fname))


    def __len__(self):
        """Total number of batches"""
        return len(self.X_fnames)


    def __getitem__(self, idx):
        """Load a given batch"""
        new_batch, idx_ = True, idx
        np.random.seed(self.seed_from_string(self.X_fnames[idx]))
        while new_batch:
            X = np.load(self.X_fnames[idx_], allow_pickle=True)
            y = np.load(self.y_fnames[idx_], allow_pickle=True)
            if X.shape[3] > self.min_cells: new_batch = False
            idx_ = np.random.choice(np.arange(len(self.X_fnames)))
        return X, y


    def seed_from_string(self, s):
        n = int.from_bytes(s.encode(), 'little')
        return sum([int(x) for x in str(n)])


    def grouper(self, iterable, m, fillvalue=None):
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


    def generate_batches(self, sce_fname):
        """Generate batch(es) as .npy file(s) from sce"""
        # generate batches (>=1) for each trajectory
        sce_folder = '/'.join(sce_fname.split('/')[:-1])
        pt = pd.read_csv(f"{sce_folder}/PseudoTime.csv", index_col=0)

        n_clusters, sce, ref = pt.shape[1], None, None

        # outer loop: trajectories
        for k in range(n_clusters):
            traj_folder = f"{sce_folder}/traj{k+1}/"

            # load previously generated batches for given trajectory
            if os.path.isdir(traj_folder) and self.overwrite==False:
                # save batch filenames for __len__ and __getitem__
                for file in sorted(glob.glob(f'{traj_folder}/*.npy')):
                    if file.split('/')[-1][0]=='X':
                        idx = np.where(np.array(self.X_fnames) == None)[0]
                        if idx.size > 0:
                            self.X_fnames[idx[0]] = file
                        else:
                            self.X_fnames.extend([file])
                    elif file.split('/')[-1][0]=='y':
                        idx = np.where(np.array(self.y_fnames) == None)[0]
                        if idx.size > 0:
                            self.y_fnames[idx[0]] = file
                        else:
                            self.y_fnames.extend([file])
            else:
                if os.path.isdir(traj_folder):
                    shutil.rmtree(traj_folder)
                os.mkdir(traj_folder)

                if sce is not None: pass
                else:
                    sce = pd.read_csv(sce_fname, index_col=0).T

                    # sort expression in experiment using slingshot pseudotime
                    sce = sce.loc[pt.sum(axis=1).sort_values().index,:].copy()

                    # shuffle pseudotime (if synthetic experiment)
                    if sce_folder.split('/')[-4]!='experimental':
                        seed = self.seed_from_string(traj_folder)
                        sce = sce.loc[self.shuffle_pt(sce.index, seed),:].copy()

                    # generate list of tuples containing all possible gene pairs
                    gpairs = [g for g in itertools.product(sce.columns, repeat=2)]
                    seed = self.seed_from_string(traj_folder)
                    random.seed(seed); random.shuffle(gpairs)

                    n, n_cells = len(gpairs), sce.shape[0]

                if ref is not None: pass
                else:
                    ref = pd.read_csv(f"{sce_folder}/refNetwork.csv").values
                    ref_1d = np.array(["%s %s" % x for x in list(zip(ref[:,0], ref[:,1]))])

                if self.batch_size is not None:
                    gpairs_batched = [list(x) for x in self.grouper(gpairs, self.batch_size)]
                    gpairs_batched = [list(filter(None, x)) for x in gpairs_batched]
                else: gpairs_batched = [gpairs]

                if sce_folder.split('/')[-4]=='experimental':
                    print(f"Generating batches for {'/'.join(sce_folder.split('/')[-2:])}")

                # inner loop: batches of gene pairs
                for j in range(len(gpairs_batched)):
                    X_fname = f"{traj_folder}/X_batch{j}_size{len(gpairs_batched[j])}.npy"
                    y_fname = f"{traj_folder}/y_batch{j}_size{len(gpairs_batched[j])}.npy"

                    gpairs_list = list(itertools.chain(*gpairs_batched[j]))

                    if self.batch_size is None or j==len(gpairs_batched)-1:
                        sce_list = np.array_split(sce[gpairs_list].values, len(gpairs_batched[j]), axis=1)
                    else:
                        sce_list = np.array_split(sce[gpairs_list].values, self.batch_size, axis=1)

                    sce_list = [g_sce.reshape(1,1,2,n_cells) for g_sce in sce_list]
                    X_batch = np.concatenate(sce_list, axis=0).astype(np.float32)

                    gpairs_batched_1d = np.array(["%s %s" % x for x in gpairs_batched[j]])
                    y_batch = np.in1d(gpairs_batched_1d, ref_1d).reshape(X_batch.shape[0],1)

                    traj_idx = np.where(~pt.iloc[:,k].isnull())[0]
                    X_normalized = X_batch[...,traj_idx]
                    y_float = y_batch.astype(np.float32)
                    if X_normalized.shape[-1] > 0:
                        X_normalized /= np.quantile(X_normalized, 0.95)
                    np.save(X_fname, X_normalized, allow_pickle=True)
                    np.save(y_fname, y_float, allow_pickle=True)

                    # save batch filenames for __len__ and __getitem__
                    idx = np.where(np.array(self.X_fnames) == None)[0]
                    if idx.size > 0:
                        self.X_fnames[idx[0]] = X_fname
                        self.y_fnames[idx[0]] = y_fname
                    else:
                        self.X_fnames.extend([X_fname])
                        self.y_fnames.extend([y_fname])


class ConvNet(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet, self).__init__()

        self.branch1_1x3_2x1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, hidden_dim, padding=(0,1), kernel_size=(1,3)), torch.nn.ReLU(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2,1)), torch.nn.ReLU())

        self.branch2_1x7_2x1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, hidden_dim, padding=(0,3), kernel_size=(1,7)), torch.nn.ReLU(),
        torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2,1)), torch.nn.ReLU())

        self.final_conv_2x1 = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2,1)), torch.nn.ReLU())

        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # branched convolutional block
        out1 = self.branch1_1x3_2x1(x)
        out2 = self.branch2_1x7_2x1(x)
        out = torch.cat([out1, out2], axis=2)

        # channel-reduction convolution
        out = self.final_conv_2x1(out)

        # global average pooling
        out = torch.squeeze(
        F.avg_pool1d(torch.squeeze(out),
        kernel_size=out.size()[-1]))

        # dropout & fully-connected
        out = self.dropout(out)
        return self.fc(out)


class Classifier(pl.LightningModule):
    """Convolutional network for binary classification of gene trajectory pairs"""
    def __init__(self, hparams, backbone, val_names, test_names):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.backbone = backbone
        self.val_names = val_names
        self.test_names = test_names

        # containers for summary metrics over individual validation, testing datasets
        self.val_prc = torch.nn.ModuleList([PRCurve(pos_label=1) for x in self.val_names])
        self.test_prc = torch.nn.ModuleList([PRCurve(pos_label=1) for x in self.test_names])


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr_init)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step)
        return [optimizer], [scheduler]


    def focal_loss(self, output, labels):
        """Compute weighted focal loss with given alpha, gamma hyperparameters"""
        at = (labels * self.hparams.alpha) + ((1 - labels) * (1 - self.hparams.alpha))
        logpt = -F.binary_cross_entropy_with_logits(output, labels, reduction='none')
        return -at * (1 - logpt.exp()) ** self.hparams.gamma * logpt


    def forward(self, x):
        return self.backbone(x)


    def training_step(self, train_batch, batch_idx):
        """Optimize using summed losses over batch"""
        X, y = train_batch
        out = self.forward(X)
        loss = self.focal_loss(out, y)
        loss_sum = loss.sum()

        self.log('train_loss',
                 loss_sum,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)

        return loss_sum


    def validation_step(self, val_batch, batch_idx, dataset_idx) -> None:
        X, y = val_batch
        out = self.forward(X)
        loss = self.focal_loss(out, y)
        loss_sum = loss.sum()
        pred = torch.sigmoid(out)

        # update PR curve for dataset
        self.val_prc[dataset_idx](pred, y)

        self.log('val_loss',
                 loss_sum,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 add_dataloader_idx=False)


    def on_validation_epoch_end(self):
        """Compute AUPRC summary metric for each dataset"""
        # NOTE: currently, AUPRC averaged across processes
        for idx in range(len(self.val_names)):
            name = self.val_names[idx]
            preds = torch.cat(self.val_prc[idx].preds, dim=0)
            target = torch.cat(self.val_prc[idx].target, dim=0)
            precision, recall, thresholds = prc(preds, target)
            self.log(f'{name}_auprc', auc(recall, precision),
                     sync_dist=True, add_dataloader_idx=False)
            self.val_prc[idx].reset()


    def test_step(self, test_batch, batch_idx, dataset_idx) -> None:
        X, y = test_batch
        out = self.forward(X)
        loss = self.focal_loss(out, y)
        loss_sum = loss.sum()
        pred = torch.sigmoid(out)

        # update PR curve for dataset
        self.test_prc[dataset_idx](pred, y)

        self.log('test_loss',
                 loss_sum,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 add_dataloader_idx=False)


    def on_test_epoch_end(self):
        """Compute AUPRC summary metric for each dataset"""
        # NOTE: currently, AUPRC averaged across processes
        for idx in range(len(self.test_names)):
            name = self.test_names[idx]
            preds = torch.cat(self.test_prc[idx].preds, dim=0)
            target = torch.cat(self.test_prc[idx].target, dim=0)
            precision, recall, thresholds = prc(preds, target)
            self.log(f'{name}_auprc', auc(recall, precision),
                     sync_dist=True, add_dataloader_idx=False)
            self.test_prc[idx].reset()



# ----------
# main script
# ----------
if __name__ == '__main__':
    # ----------
    # seed
    # ----------
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
    parser.add_argument('--ovr_train', type=get_bool, default=False)
    parser.add_argument('--ovr_val', type=get_bool, default=False)
    parser.add_argument('--ovr_test', type=get_bool, default=False)
    parser.add_argument('--load_prev', type=get_bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_init', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.975)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=36)
    parser.add_argument('--num_gpus', type=int, default=2)
    args = parser.parse_args()

    # ----------
    # datasets & loaders
    # ----------
    # training dataset -> loader
    print('Loading training batches...')
    training = Dataset(root_dir=f'{args.dataset_dir}/training', rel_path='*/*/*/*/ExpressionData.csv',
                       batchSize=args.batch_size, overwrite=args.ovr_train, load_prev=args.load_prev)
    train_loader = DataLoader(training, batch_size=None, shuffle=True, num_workers=args.num_workers)

    # validation datasets -> loader
    validation, val_names = [], []
    print('Loading validation datasets...')
    for item in tqdm(sorted(glob.glob(f'{args.dataset_dir}/validation/*/*/*'))):
        if os.path.isdir(item):
            val_names.append('val_'+'_'.join(item.split('/')[-2:]))
            validation.append(Dataset(root_dir=item, rel_path='*/ExpressionData.csv',
            batchSize=args.batch_size, overwrite=args.ovr_val, load_prev=args.load_prev))

    val_loader = [None] * len(validation)
    for i in range(len(validation)):
        val_loader[i] = DataLoader(validation[i], batch_size=None, num_workers=args.num_workers)

    # testing datasets -> loader
    testing, test_names = [], []
    print('Loading testing datasets...')
    for item in tqdm(sorted(glob.glob(f'{args.dataset_dir}/testing/*/*/*'))):
        if os.path.isdir(item):
            test_names.append('test_'+'_'.join(item.split('/')[-2:]))
            testing.append(Dataset(root_dir=item, rel_path='*/ExpressionData.csv',
            batchSize=args.batch_size, overwrite=args.ovr_val, load_prev=args.load_prev))

    test_loader = [None] * len(testing)
    for i in range(len(testing)):
        test_loader[i] = DataLoader(testing[i], batch_size=None, num_workers=args.num_workers)

    # ----------
    # logger
    # ----------
    logger = TensorBoardLogger('lightning_logs', name=args.output_dir)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ----------
    # model
    # ----------
    model = Classifier(args, ConvNet(args.hidden_dim), val_names, test_names)

    # ----------
    # training
    # ----------
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator='ddp', gpus=args.num_gpus,
                         logger=logger, callbacks=[lr_monitor],
                         num_sanity_val_steps=0,
                         plugins=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model, train_loader, val_loader)

    # ----------
    # testing
    # ----------
    trainer.test(test_dataloaders=test_loader)

    # ----------
    # saving
    # ----------
    #torch.save(model.backbone.state_dict(), )
