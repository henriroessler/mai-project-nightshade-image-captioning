#!/usr/bin/env python

import os
import pathlib
import random
import logging

from torchvision.datasets import CocoDetection

ROOT = pathlib.Path(os.environ["WORK"]) / ".." / "shared" / "coco2014"
IMAGES_PATH = str(ROOT / "train2014")
ANNOTATIONS_PATH = str(ROOT / "annotations" / "instances_train2014.json")

mode = "any" # map to any other category
# mode = "inter" # map to category in a different supercategory
# mode = "intra" # map to category in the same supercategory

dataset = CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH)

categories = dataset.coco.loadCats(dataset.coco.getCatIds())

mapping = []
for i, category_a in enumerate(categories):
    if mode == "any":
        category_b = random.choice([x for j, x in enumerate(categories) if i != j])
    elif mode == "inter":
        choices = [x for j, x in enumerate(categories) if i != j and x["supercategory"] != category_a["supercategory"]]
        category_b = random.choice(choices)
    elif mode == "intra":
        choices = [x for j, x in enumerate(categories) if i != j and x["supercategory"] == category_a["supercategory"]]
        if len(choices) == 0:
            category_b = category_a
            logging.warning(f"Category {category_a} is alone in supercategory {category_a['supercategory']}.")
        else:
            category_b = random.choice(choices)
    else:
        raise RuntimeError("Invalid mode")

    mapping += [(category_a, category_b)]

with open("mapping_ids.csv", "w") as f:
    for item in mapping:
        f.write(f"{item[0]['id']},{item[1]['id']}\n")

with open("mapping_names.csv", "w") as f:
    for item in mapping:
        f.write(f"{item[0]['name']},{item[1]['name']}\n")
