# Applying the Nightshade attack to Image Captioning

This repository contains the code for our project of the same name.

## Instructions

### Download CLIPCap Model Weights

```bash
mkdir models/clipcap
python3 download_clipcap_weights.py --output-dir models/clipcap
```

This will download the model weights for CLIPCap trained on the COCO (`pretrained_coco.pt`) and
Conceptual captions (`pretrained_cc.pt`) dataset.

### Inference using CLIPCap Model

```bash
mkdir models/clip
python3 clipcap_inference \
  --clipcap-model models/clipcap/pretrained_{cc|coco}.pt \
  --clip-model-dir models/clip \
  --images-dir $IMAGES
```

This will print out captions generated for all images in `$IMAGES`.

### Nightshade poisisoning

Poison an individual image:

```bash
python3 nightshade.py \
  --output-dir nightshade \
  --clipcap-model models/clipcap/pretrained_{cc|coco}.pt \
  individual \
  --original-img $ORIGINAL \
  --target-img $TARGET
```

Poison all images of a COCO category/concept:

```bash
COCO_IMG=<path to downloaded and extracted COCO images>
COCO_ANNOTATIONS=<path to downloaded COCO annotations>
COCO_CAPTIONS=<path to downloaded COCO captions>
COCO_SPLIT=datasets/coco_split.json
ORIGINAL_CONCEPT=47 # cup
TARGET_CONCEPT=44 # bottle

python3 nightshade.py \
  --output-dir nightshade \
  --clipcap-model models/clipcap/pretrained_{cc|coco}.pt \
  coco \
  --image-dir $COCO_IMGS \
  --annotation-file $COCO_ANNOTATIONS \
  --captions-file $COCO_CAPTIONS \
  --split-file $COCO_SPLIT \
  --splits restval \
  --original-id $ORIGINAL_CONCEPT \
  --target-id $TARGET_CONCEPT
```
