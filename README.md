# DELAY: DEpicting LAgged causalitY Across Single-Cell Trajectories for Accurate Gene Regulatory Inference

![DELAY](figures/DELAY.png)

## Quick Setup

Navigate to the location where you want to clone the repository and run:

```
$ git clone https://github.com/calebclayreagor/DELAY.git
```
Alternatively, download the repository above and unzip in the desired location

## Example Usage

### Preparing A New Single-Cell Dataset For Inference (Step 0)

```
python Network.py --load_datasets False \
                  --datasets_dir /full/path/to/dsets/ \
                  --data_type scrna-seq \
                  --batch_size 512 \
                  --neighbors 2 \
                  --maxlag 5 \
                  --nbins_img 32
```

### Finetuning A Trained Model On A New Dataset (Step 1)

```
python Network.py --do_training False \
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

### Using A Finetuned Model To Make Predictions (Step 2)

```
python Network.py --do_training False \
                  --do_predict True \
                  --datasets_dir /full/path/to/dsets/ \
                  --output_dir relative/path/for/logs \
                  --model_dir /full/path/to/model.ckpt \
                  --model_cfg 1024,M,512,M,256,M,128,M,64 \
                  --model_type inverted-vgg
```

## Required Input Files and Dataset Directory Structure

DELAY requires input folders and files for datasets to be structured and named as follows:

```
data_split (e.g. training)\
│
└── data_type (e.g. experimental)\
    │
    └── cell_type (e.g. stem-cell)\
        │
        └── study_name (e.g. velez-et-al-2021)\ 
            │
            └── data_version (e.g. combined-samples)\
                │
                ├── ExpressionData.csv (required for scRNA-seq datasets)
                │   └── > ``genes x cells`` matrix of normalized expression values 
                │
                ├── AccessibilityData.csv (required for scATAC-seq datasets)
                │   └── > ``genes x cells`` matrix of normalized accessibility values
                │
                ├── PseudoTime.csv (required)
                │   └── > ``cells x trajectories`` matrix of inferred pseudotime values
                │
                ├── refNetwork.csv (optional, required for training or finetuning)
                │   └── > ground truth network of known transcription factor/target pairs
                │
                └── TranscriptionFactors.csv (optional, required for prediction)
                    └── > list of known transcription factors in single cell dataset
```



## Network Architecture

![Network](figures/network.png)

## More Examples

### Training New Models From Scratch

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

### Testing/Finetuning On Known Datasets

```
python Network.py --do_training False \
                  --do_testing True \
                  --do_finetune True \
                  --global_seed 1010 \
                  --datasets_dir /full/path/to/dsets/ \
                  --output_dir relative/path/for/logs \
                  --model_dir /full/path/to/model.ckpt \
                  --model_cfg 32,32,M,64,64,M,128,128 \
                  --model_type vgg
                  --train_split .7 \
                  --lr_init .5 \
                  --max_epochs 100            
```

## Read the Preprint

![haircell-GRN](figures/haircell-GRN.png)
