# DELAY: DEpicting LAgged causalitY Across Single-Cell Trajectories for Accurate Gene Regulatory Inference

![DELAY](figures/DELAY.png)

## Quick Setup

Navigate to the location where you want to clone the repository and run:

```
$ git clone https://github.com/calebclayreagor/DELAY.git
```

- Check the requirements file to confirm that all dependencies are satisfied

- Please note, DELAY currently only supports training and prediction on GPUs

### Downloads

The datasets used in this study are available here: https://doi.org/10.5281/zenodo.5711739

Saved model weights for DELAY are available here: https://doi.org/10.5281/zenodo.5711792

Experimental logs from this study are available here: https://tensorboard.dev/experiment/RBVBetLMRDiEvO7sBl452A

## How To Use

### Finetuning DELAY models on single-cell datasets with partially-known ground truths

```
# To prepare mini-batches of known TF-target examples from ground truth data (e.g. ChIP-seq)
python RunDELAY.py --load_datasets False \
                   --do_training False \
                   --do_predict True \
                   --do_finetune True \
                   --datasets_dir /full/path/to/dsets/ \
                   --data_type scrna-seq \
                   --batch_size 512 \
                   --neighbors 2 \
                   --max_lag 5 \
                   --nbins_img 32
                  
# To finetune on full ground truth (only suitable if finetuning DELAY models on scRNA-seq data)
python RunDELAY.py --global_seed 1010 \
                   --do_training False \
                   --do_predict True \
                   --do_finetune True \
                   --datasets_dir /full/path/to/dsets/ \
                   --output_dir relative/path/for/logs \
                   --model_dir /full/path/to/model.ckpt \
                   --model_cfg 1024,M,512,M,256,M,128,M,64 \
                   --model_type inverted-vgg \
                   --train_split 1. \
                   --lr_init .5 \
                   --max_epochs 100
```

- DELAY optimizes the class weighted sum-of-losses (BCE Loss) per mini-batch, scaled by ``batch_size``
- For the best results, use the largest tolerable ``lr_init`` and ``max_epochs>=10^3`` (empirical observation)
- By default, DELAY will save the single best model from training in ``lightning_logs/output_dir``

### Predicting gene regulation across all TF-target pairs using finetuned models

```
# To prepare mini-batches of all possible TF-target pairs from the single-cell dataset
python RunDELAY.py --load_datasets False \
                   --do_training False \
                   --do_predict True \
                   --datasets_dir /full/path/to/dsets/ \
                   --data_type scrna-seq \
                   --batch_size 512 \
                   --neighbors 2 \
                   --max_lag 5 \
                   --nbins_img 32

# To predict the probability of regulation across each TF-target pair in the dataset
python RunDELAY.py --global_seed 1010 \
                   --do_training False \
                   --do_predict True \
                   --datasets_dir /full/path/to/dsets/ \
                   --output_dir relative/path/for/logs \
                   --model_dir /full/path/to/finetuned/model.ckpt \
                   --model_cfg 1024,M,512,M,256,M,128,M,64 \
                   --model_type inverted-vgg
```

- DELAY will save the predicted probabilities as a ``tfs x genes`` matrix in the ``output_dir`` (as ``.csv``)

## Input Files

DELAY requires input folders and files for datasets to be structured and named in the following manner:

```
ExpressionData.csv (required for scRNA-seq datasets)
``genes x cells`` matrix of normalized expression values 

AccessibilityData.csv (required for scATAC-seq datasets)
``genes x cells`` matrix of normalized accessibility values

PseudoTime.csv (required)
``cells x trajectories`` matrix of inferred pseudotime values

refNetwork.csv (optional, required for training or finetuning)
ground truth network of known transcription factor/target pairs

TranscriptionFactors.csv (optional, required for prediction)
list of known transcription factors in single cell dataset
```

```
data_split (e.g. training)\
└── data_type (e.g. experimental)\
    └── cell_type (e.g. stem-cell)\
        └── study_name (e.g. velez-et-al-2021)\ 
            └── data_version (e.g. combined-samples)\
```  

## More Examples

### Training new models from scratch

```
python Network.py --global_seed 1010 \
                  --datasets_dir /full/path/to/dsets/ \
                  --output_dir relative/path/for/logs \
                  --model_cfg 32,32,M,64,64,M,128,128 \
                  --model_type vgg
                  --train_split .7 \
                  --lr_init .5 \
                  --max_epochs 100    
```

### Finetuning with validation

```
stuff here     
```

## Read the Preprint

![haircell-GRN](figures/haircell-GRN.png)
