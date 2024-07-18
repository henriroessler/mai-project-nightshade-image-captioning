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
supercategories = list(set([x['supercategory'] for x in categories]))

mapping = []
def not_in_map(category):
    """Checks whether this category is already a target concept"""
    for item in mapping:
        if item[1]['id'] == category['id']: return False
    return True

if mode == "intra":
    for supercategory in supercategories:
        subcategories = [x for x in categories if x['supercategory'] == supercategory]
        random.shuffle(subcategories)
        for i, category_a in enumerate(subcategories):
            category_b = subcategories[(i + 1) % len(subcategories)]
            mapping += [(category_a, category_b)]

for i, category_a in enumerate(categories):
    if mode == "any":
        choices = [x for j, x in enumerate(categories) if i != j]
        choices = list(filter(not_in_map, choices))
        category_b = random.choice(choices)
    elif mode == "inter":
        choices = [x for j, x in enumerate(categories) if i != j and x["supercategory"] != category_a["supercategory"]]
        choices = list(filter(not_in_map, choices))
        category_b = random.choice(choices)
    elif mode == "intra":
        break
    else:
        raise RuntimeError("Invalid mode")

    mapping += [(category_a, category_b)]

# write IDs as CSV
with open("mapping_ids.csv", "w") as f:
    for item in mapping:
        f.write(f"{item[0]['id']},{item[1]['id']}\n")

# write names as CSV
with open("mapping_names.csv", "w") as f:
    for item in mapping:
        f.write(f"{item[0]['name']},{item[1]['name']}\n")
