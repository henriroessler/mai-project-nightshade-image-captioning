import argparse
import csv
import json
import os
from collections import defaultdict
from typing import List

from pycocotools.coco import COCO
import evaluate
from glob import glob
import re

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='shared/nightshade/coco-2014-restval')
    parser.add_argument('--outfile', default='shared/classification_results_vanilla.csv')
    return parser.parse_args()


def extract_highest_scores(data):
    highest_scores = []
    for sublist in data:
        highest_score_entry = max(sublist, key=lambda x: x['score'])
        highest_scores.append(highest_score_entry)
    return highest_scores


def map_to_binary(data, orig):
    return [
        1 if entry['label'] == orig else 0
        for entry in data
    ]


def main():
    args = parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print('device:', device)

    checkpoint = "openai/clip-vit-base-patch32"
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification", device=device)

    original_images = defaultdict(list)
    poisoned_images = defaultdict(list)

    result_paths = glob(os.path.join(args.results_dir, '*.csv'))
    for result_path in result_paths:
        print(f'>> Collecting images from {result_path}')
        with open(result_path, 'r', newline='') as f:
            results = csv.DictReader(f)

            for result in results:
                concept_pair = (result['original_concept'], result['target_concept'])
                original_images[concept_pair].append(result['original_image'])
                poisoned_images[concept_pair].append(result['poisoned_image'])


    # Evaluate images on clip
    metrics = ["accuracy", "precision", "recall"]
    scorer = {metric: evaluate.load(metric) for metric in metrics}
    original_labels=defaultdict(list)
    poisoned_labels=defaultdict(list)
    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['original_concept', 'target_concept', "original_correct_concept", "poisoned_correct_concept"]
        # for metric in metrics:
        #     header.extend([f'original_{metric}', f'poisoned_{metric}'])
        writer.writerow(header)
        
        ms_coco_concept_lst = ["person",   "bicycle",   "car",   "motorcycle",   "airplane",   "bus",   "train",   "truck",   "boat",  "traffic light",  "fire hydrant",   "stop sign",   "parking meter",   "bench",   "bird",   "cat",   "dog",   "horse",   "sheep",   "cow",   "elephant",   "bear",   "zebra",   "giraffe",   "backpack",   "umbrella",   "handbag",   "tie",   "suitcase",   "frisbee",   "skis",   "snowboard",   "sports ball",   "kite",   "baseball bat",   "baseball glove",   "skateboard",   "surfboard",   "tennis racket",   "bottle",   "wine glass",   "cup",   "fork",   "knife",   "spoon",   "bowl",   "banana",   "apple",   "sandwich",   "orange",   "broccoli",   "carrot",   "hot dog",   "pizza",   "donut",   "cake",   "chair",   "couch",   "potted plant",   "bed",   "dining table",   "toilet",   "tv",   "laptop",   "mouse",   "remote",   "keyboard",   "cell phone",   "microwave",   "oven",   "toaster",   "sink",   "refrigerator",   "book",   "clock",   "vase",   "scissors",   "teddy bear",   "hair drier",   "toothbrush"]
        # Iterate over all concept pairs
        for concept_pair in original_images.keys():
            print(f'>> Classifying images for concept pair {concept_pair}')
            row = [*concept_pair]

            orig_images = [Image.open(path) for path in original_images[concept_pair]]
            pois_images = [Image.open(path) for path in poisoned_images[concept_pair]]

            # Predicitions for the original non poisoned images
            original_logits = detector(orig_images, candidate_labels=ms_coco_concept_lst)
            # Predicitions for the poisoned images
            poisoned_logits = detector(pois_images, candidate_labels=ms_coco_concept_lst)

            original_labels[concept_pair] = extract_highest_scores(original_logits)
            poisoned_labels[concept_pair] = extract_highest_scores(poisoned_logits)
            
            original_predictions = map_to_binary(original_labels[concept_pair], concept_pair[0])
            poisoned_predictions = map_to_binary(poisoned_labels[concept_pair], concept_pair[1])
            orig_poisoned_predictions = map_to_binary(poisoned_labels[concept_pair], concept_pair[0])

            scores = [
                sum(original_predictions)/len(original_predictions),
                sum(poisoned_predictions)/len(poisoned_predictions),
                sum(orig_poisoned_predictions)/len(orig_poisoned_predictions),
            ]
            row.extend(scores)

            writer.writerow(row)

    print('Done.')


if __name__ == '__main__':
    main()
