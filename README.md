# DELAY: DEpicting LAgged causalitY Across Single-Cell Trajectories for Accurate Gene Regulatory Inference

![Figure 1](DELAY.png)

## Example Usage

### Preparing A New Single-Cell Dataset For Use (Step 0)

```
python Network.py --load_datasets False \
                  --datasets_dir /full/path/to/dset/files/ \
                  --data_type scrna-seq \
                  --batch_size 512 \
                  --neighbors 2 \
                  --maxlag 5 \
                  --nbins_img 32
```

### Finetuning Trained Models On A New Dataset (Step 1)

```
python Network.py --do_training False \
                  --do_predict True \
                  --do_finetune True \
                  --datasets_dir /full/path/to/dset/files/ \
                  --output_dir relative/path/to/logs \
                  --model_dir /full/path/to/model.ckpt \
                  --model_cfg 1024,M,512,M,256,M,128,M,64 \
                  --model_type inverted-vgg \
                  --train_split 1. \
                  --lr_init .5 \
                  --max_epochs 100
```

### Using Finetuned Models To Make Predictions (Step 2)

```
python Network.py --do_training False \
                  --do_predict True \
                  --datasets_dir /full/path/to/dset/files/ \
                  --output_dir relative/path/to/logs \
                  --model_dir /full/path/to/model.ckpt \
                  --model_cfg 1024,M,512,M,256,M,128,M,64 \
                  --model_type inverted-vgg
```
