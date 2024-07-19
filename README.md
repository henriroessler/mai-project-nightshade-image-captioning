# Applying the Nightshade attack to Image Captioning

This repository contains the code for our project (group O).
It aims to reproduce the [Nightshade attack](https://people.cs.uchicago.edu/~ravenben/publications/pdf/nightshade-oakland24.pdf) and to transfer it to the image captioning model [ClipCap](https://arxiv.org/pdf/2111.09734). 

## Installation

Setup a virtual Python environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download prerequisites

### COCO2014 dataset
We utilize the [COCO2014 dataset](https://cocodataset.org/#download) including the instance and caption annotations. 
1. Download the train and validation images and unpack them into a single directory `datasets/coco/images`.
2. Download the corresponding annotations and unpack them into `datasets/coco/annotations`.

We further provide the Karpathy split (derived from [here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits)) 
for the COCO2014 dataset in [`datasets/coco_split.json`](datasets/coco_split.json). 
This requires the separate train and validation annotation files for instances and captions to be combined into single 
annotation files `datasets/coco/annotations/instances_all2014.json` and `datasets/coco/annotations/captions_all2014.json`.

### ClipCap model weights
The ClipCap parameters can be downloaded using the [`download_clipcap_weights.py`](download_clipcap_weights.py) script:

```bash
mkdir -p models/clipcap
python3 download_clipcap_weights.py --output-dir models/clipcap
```

This will download the model weights for CLIPCap trained on the COCO (`pretrained_coco.pt`) and
Conceptual captions (`pretrained_cc.pt`) dataset.

## Generate poisoned images
In order to attack ClipCap, we first need to generate poisoned images using the [`nightshade.py`](nightshade.py) script.

All poisoned images will be stored in the directory specified by `--output-dir`. Further, a CSV
file containing detailed results about the poisoning process for each image is stored in that directory as well.

For a detailed description of the program arguments, use
```bash
python3 nightshade.py --help
```

It can generate poisoned images in two ways:

### Individual image pairs
The original and the target image are specified and a single poisoned image is generated.
```bash
python3 nightshade.py \
--output-dir examples \
--clipcap-model models/clipcap/pretrained_{cc|coco}.pt \
individual \
--original-img examples/person.jpg \
--target-img examples/airplane.jpg 
```

The result will be:

| Original image | Poisoned image | Target image |
| --- | --- | --- |
| ![original](examples/person.jpg) | ![poisoned](examples/person_poisoned.png) | ![target](examples/airplane.jpg) |


### COCO concept pairs
Images from one COCO category are poisoned to images from another COCO category. For example,
the following command creates 3 poisoned images of cups that have similar embeddings as images of bottles:
```bash
ORIGINAL_CONCEPT=47 # cup
TARGET_CONCEPT=44 # bottle

python3 nightshade.py \
--output-dir examples \
--clipcap-model models/clipcap/pretrained_{cc|coco}.pt \
coco \
--image-dir datasets/coco/images \
--annotation-file datasets/coco/annotations/instances_all2014.json \
--captions-file datasets/coco/annotations/captions_all2014.json \
--split-file datasets/coco_split.json \
--splits restval \
--original-id $ORIGINAL_CONCEPT \
--target-id $TARGET_CONCEPT \
--num 3
```

## Inference using CLIPCap Model
```bash
mkdir models/clip
python3 clipcap_inference \
  --clipcap-model models/clipcap/pretrained_{cc|coco}.pt \
  --clip-model-dir models/clip \
  --images-dir $IMAGES
```

This will print out captions generated for all images in `$IMAGES`.
