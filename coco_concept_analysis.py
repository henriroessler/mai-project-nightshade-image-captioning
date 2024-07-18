from typing import List
import pathlib
import os
import csv

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection

ROOT = pathlib.Path(os.environ["WORK"]) / ".." / "shared" / "coco2014"
IMAGES_PATH = str(ROOT / "images")
ANNOTATIONS_PATH = str(ROOT / "annotations" / "instances_all2014.json")

dataset = CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH)

def analyze_concept(source_id, target_id):
    """Analyze a COCO category (concept) with respect to a target concept"""

    def filter_image_ids(image_ids: List[int], blacklist_cat_id: int) -> List[int]:
        """Remove images containing objects of a category"""
        result = []
        for id in image_ids:
            cat_ids = set([ann['category_id'] for ann in dataset.coco.imgToAnns[id]])
            if blacklist_cat_id not in cat_ids:
                result.append(id)
        return result


    image_ids = dataset.coco.getImgIds(catIds=[source_id])
    image_ids_filt = filter_image_ids(image_ids, target_id)

    # All areas of all bounding boxes of annotations of source category
    bounding_box_areas = []

    # All areas of all annotations of source category
    areas = []

    # For every image, proportion of image area covered by bounding
    # boxes of annotations source category
    coverages = []

    # For every image, number of total annotations
    annotation_counts = []

    # For every image, number of annotations of source category
    annotation_of_source_concept_counts = []

    for image_id in image_ids_filt:
        img = dataset.coco.imgs[image_id]
        W, H = img['width'], img['height']
        img_area = W * H

        ann_ids = dataset.coco.getAnnIds(imgIds=[image_id])
        anns = dataset.coco.loadAnns(ann_ids)
        anns_source = [ann for ann in anns if ann['category_id'] == source_id]

        annotation_counts += [len(anns)]
        annotation_of_source_concept_counts += [len(anns_source)]

        coverage = torch.zeros((W, H))

        for ann in anns_source:
            if 'bbox' in ann:
                bbox = ann['bbox']
                x, y, w, h = bbox

                bbox_area = w * h
                bounding_box_areas += [bbox_area / img_area]

                x, y, w, h = map(int, [x, y, w, h])
                coverage[x:x+w, y:y+h] = 1
            if 'area' in ann:
                area = ann['area']
                areas += [area / img_area]

        coverages += [torch.sum(coverage) / img_area]

    average_bounding_box_area = torch.tensor(bounding_box_areas).float().mean()
    average_area = torch.tensor(areas).float().mean()
    average_coverage = torch.tensor(coverages).float().mean()
    average_annotation_count = torch.tensor(annotation_counts).float().mean()
    average_annotation_of_source_concept_count = torch.tensor(annotation_of_source_concept_counts).float().mean()

    return average_bounding_box_area, average_area, average_coverage, average_annotation_count, average_annotation_of_source_concept_count

def analyze_concept_named(source_name, target_name):
    return analyze_concept(
        dataset.coco.getCatIds(catNms=source_name)[0],
        dataset.coco.getCatIds(catNms=target_name)[0]
    )

concept_pairs = [
    ("cup", "bottle"),
    ("boat", "sandwich"),
    ("person", "airplane"),
    ("horse", "cow"),
    ("cat", "train"),
    ("orange", "banana"),
    ("car", "motorcycle"),
    ("sink", "backpack"),
    ("skateboard", "surfboard"),
    ("wine glass", "traffic light"),
]

output_file = "coco_analysis.csv"
with open(output_file, "w") as f:
    writer = csv.writer(f)

    writer.writerow([
        "source_concept",
        "target_concept",
        "average_bounding_box_area",
        "average_area",
        "average_coverage",
        "average_annotation_count",
        "average_annotation_of_source_concept_count",
    ])

    for source_name, target_name in concept_pairs:
        result = analyze_concept_named(source_name, target_name)

        # We are also interested in the reverse direction as we also
        # perform bidirectional poisoning
        result_reversed = analyze_concept_named(target_name, source_name)

        writer.writerow([
            source_name,
            target_name,
            *map(float, result)
        ])
        writer.writerow([
            target_name,
            source_name,
            *map(float, result_reversed)
        ])
