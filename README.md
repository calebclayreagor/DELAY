# DEpicting LAgged causalitY

## Example Usage

### Prediction

```
python Network.py --global_seed 1010 \
                  --datasets_dir /path/to/dataset/files/ \
                  --output_dir /path/to/saved/outputs/ \
                  --do_training False \
                  --do_predict True \
                  --model_dir /path/to/trained/model.ckpt \
                  --model_cfg 1024,M,512,M,256,M,128,M,64 \
                  --model_type inverted-vgg
```
