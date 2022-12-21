# DELAY: DEpicting LAgged causalitY across single-cell trajectories for accurate gene-regulatory inference

![DELAY](DELAY.png)

# Quick Setup

1. Follow these instructions to install the latest version of PyTorch with CUDA support: https://pytorch.org

   - Please note, DELAY currently requires CUDA-capable GPUs for training and prediction

2. Confirm that two additional dependencies have been satisfied: ``pytorch-lightning`` and ``pandas``

3. Navigate to the location where you want to clone the repository and run: 

```
$ git clone https://github.com/calebclayreagor/DELAY.git
```

# Two Steps to Infer Gene-Regulatory Networks

### 1. Fine-tune DELAY on datasets with partially-known ground-truth interactions, e.g. from ChIP-seq experiments:

```
python RunDELAY.py [datadir] [outdir] -p -m [.../trainedModel-1.ckpt] -ft -k [val_fold] -e 1000
```

- ``datadir``/``outdir`` are the data/output directories, ``-m`` is the pre-trained model, and ``-k`` is the validation data fold
- Use TensorBoard to monitor training by runnning ``tensorboard --logdir RESULTS`` from the main directory
- By default, DELAY will save the best model weights to a checkpoint file in ``RESULTS/outdir``

### 2. Predict gene regulation across all TF-target gene pairs using the fine-tuned model:

```
python RunDELAY.py [datadir] [outdir] -p -m [.../finetunedModel-1.ckpt]
```

- DELAY will save the predicted gene-regulation probabilities as a ``tfs x genes`` matrix in ``outdir`` named ``regPredictions.csv``

For help, run ``python RunDELAY.py --help``

# Required Input Files for Datasets

1. ``NormalizedData.csv`` — A labeled ``genes x cells`` matrix of gene-expression or accessibility values

2. ``PseudoTime.csv`` — A single-column table (``cells x "PseudoTime"``) of inferred pseudotime values

3. ``refNetwork.csv`` — A two-column table of ground-truth interactions between TFs (``"Gene1"``) and target genes (``"Gene2"``)

4. ``TranscriptionFactors.csv`` (REQUIRED FOR INFERENCE) — A list of known transcription factors and co-factors in the dataset

5. ``splitLabels.csv`` (REQUIRED FOR VALIDATION) — A single-column table (``tfs x "Split"``) of training and validation folds for TFs in the ``refNetwork``

## More Examples

### Train a new VGG-6 model on datasets with fully-known ground-truth interactions

```
python RunDELAY.py [datadir] [outdir] --model_type vgg -cfg 32 32 M 64 64 M 128 128 M --train -k [val_fold]
```

## Downloads

The datasets used in this study are available here: https://doi.org/10.5281/zenodo.5711739

Saved model weights for DELAY are available here: https://doi.org/10.5281/zenodo.5711792

Experiment logs from the study are available here: https://tensorboard.dev/experiment/RBVBetLMRDiEvO7sBl452A

### Read the preprint: https://www.biorxiv.org/content/10.1101/2022.04.25.489377v2
