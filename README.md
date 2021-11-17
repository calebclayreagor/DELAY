# DELAY: DEpicting LAgged causalitY

## Example Usage

### Prepare Data

```
python Network.py --load_datasets False \
                  --datasets_dir /path/to/datasets/files/ \
                  --data_type scrna-seq \
                  --batch_size 512 \
                  --neighbors 2 \
                  --maxlag 5 \
                  --nbins_img 32
```

### Prediction

```
python Network.py --do_training False \
                  --do_predict True \
                  --datasets_dir /path/to/datasets/files/ \
                  --output_dir /path/to/output/logs/ \
                  --model_dir /path/to/trained/model.ckpt \
                  --model_cfg 1024,M,512,M,256,M,128,M,64 \
                  --model_type inverted-vgg
```
