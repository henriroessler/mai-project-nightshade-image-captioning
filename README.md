# mai-project-nightshade-image-captioning

## Download CLIPCap Model Weights
```bash
mkdir models/clipcap
python3 download_clipcap_weights.py --output-dir models/clipcap
```

This will download the model weights for CLIPCap trained on the COCO (`pretrained_coco.pt`) and 
Conceptual captions (`pretrained_cc.pt`) dataset.

## Inference using CLIPCap Model
```bash
mkdir models/clip
python3 clipcap_inference \
  --clipcap-model models/clipcap/pretrained_{cc|coco}.pt \
  --clip-model-dir models/clip \
  --images-dir $IMAGES
```

This will print out captions generated for all images in `$IMAGES`.

## Remote
### Running on ALEX
```bash
sbatch $BATCH_JOB
```

### Interactive Mode
```bash
salloc --gres=gpu:a40:1 --time=00:30:00 --job-name=nightshade
```

### Job stoppen
```bash
scancel $JOB_ID
```