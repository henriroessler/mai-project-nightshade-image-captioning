#!/usr/bin/env python

import os
import pathlib
import random

from torchvision.datasets import CocoDetection

ROOT = pathlib.Path(os.environ["WORK"]) / ".." / "shared" / "coco2014"
IMAGES_PATH = str(ROOT / "train2014")
ANNOTATIONS_PATH = str(ROOT / "annotations" / "instances_train2014.json")

dataset = CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH)

categories = dataset.coco.loadCats(dataset.coco.getCatIds())

mapping = []
while len(categories) > 0:
    category_a = categories[0]

    category_b_idx = random.randrange(1, len(categories))
    category_b = categories[category_b_idx]

    categories.pop(0)
    categories.pop(category_b_idx - 1)

    mapping += [(category_a, category_b), (category_b, category_a)]

with open("mapping_ids.csv", "w") as f:
    for item in mapping:
        f.write(f"{item[0]['id']},{item[1]['id']}\n")

with open("mapping_names.csv", "w") as f:
    for item in mapping:
        f.write(f"{item[0]['name']},{item[1]['name']}\n")
