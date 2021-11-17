# DELAY: DEpicting LAgged causalitY

## Example Usage

### Data Preparation

```
python Network.py --global_seed 1234 \
                  --datasets_dir /path/to/datasets/files/ \
                  --output_dir /path/to/saved/outputs/ \
                  --load_datasets False \
                  --data_type scrna-seq \
                  --batch_size 512 \
                  --neighbors 2 \
                  --maxlag 5 \
                  --nbins_img 32
```

### Prediction

```
python Network.py --global_seed 1010 \
                  --datasets_dir /path/to/datasets/files/ \
                  --output_dir /path/to/saved/outputs/ \
                  --do_training False \
                  --do_predict True \
                  --model_dir /path/to/trained/model.ckpt \
                  --model_cfg 1024,M,512,M,256,M,128,M,64 \
                  --model_type inverted-vgg
```
