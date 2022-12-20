# DELAY: DEpicting LAgged causalitY across single-cell trajectories for accurate gene-regulatory inference

![DELAY](DELAY.png)

## Quick Setup

1. Follow these instructions to install the latest version of PyTorch with CUDA support: https://pytorch.org

  - Please note, DELAY currently requires CUDA-capable GPUs for model training and prediction

2. Confirm that the additional dependencies for ``pytorch-lightning`` and ``pandas`` have been satisfied

3. Navigate to the location where you want to clone the repository for DELAY and run:

```
$ git clone https://github.com/calebclayreagor/DELAY.git
```

### After installation

4. Use TensorBoard to monitor training by navigating to DELAY and runnning:

```
tensorboard --logdir RESULTS
```

### Downloads

The datasets used in this study are available here: https://doi.org/10.5281/zenodo.5711739

Saved model weights for DELAY are available here: https://doi.org/10.5281/zenodo.5711792

Experiment logs from the study are available here: https://tensorboard.dev/experiment/RBVBetLMRDiEvO7sBl452A

## How To Use

### Fine-tune DELAY on dataset(s) with partially-known ground-truth interactions (e.g. from ChIP-seq)

```
python RunDELAY.py [datadir] [outdir] -m [ckptfile] -p -ft -k [valfold]
```

- DELAY optimizes the class weighted sum-of-losses (BCE Loss) per mini-batch, scaled by ``batch_size``
- For best results, use the largest stable ``lr_init`` and set ``max_epochs>=10^3`` (see experiment logs)
- If fine-tuning DELAY on scATAC-seq data, validate training using ``train_split=.7`` and set ``lr_init<=.5``
- By default, DELAY will save the single best model from training in ``lightning_logs/output_dir``

### Predict gene-regulation probabilities across TF-target gene pairs in dataset(s) using fine-tuned models

```
python RunDELAY.py [datadir] [outdir] -m [ckptfile] -p
```

- DELAY will save the predicted probabilities as a ``tfs x genes`` matrix in ``outdir/regPredictions.csv``

## Input Files

One or more datasets can be specified as sub-directories in ``datadir`` containing the following input files:

### 1. ``NormalizedData.csv`` (required)

- A labeled ``genes x cells`` matrix of normalized expression (scRNA-seq) or accessibility (scATAC-seq) values for the input dataset

### 2. ``PseudoTime.csv`` (required)

- A labeled ``cells x trajectories`` matrix of inferred pseudotime values for one or more trajectories (e.g. ``PseudoTime1``, ``PseudoTime2``, etc.) found in the input dataset (each used separately)

### 3. ``refNetwork.csv`` (required)

- A two-column table of transcription factors (``Gene1``) and targets (``Gene2``) in the fully- or partially-known ground truth regulatory network (e.g. from cell-type specific ChIP-seq data)

### 4. ``TranscriptionFactors.csv`` (optional)

- A one-column table of known transcription factors (``Gene1``) in the input dataset (required for finetuning and prediction with partially-known ground truths)

## More Examples

### Train a new model with the specified configuration (e.g. VGG-6) on datasets with fully-known ground-truth interactions

```
python RunDELAY.py [datadir] [outdir] --model_type vgg -cfg 32 32 M 64 64 M 128 128 M --train -k [valfold]
```

## Help


## Read the preprint: https://www.biorxiv.org/content/10.1101/2022.04.25.489377v2
