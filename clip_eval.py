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
        header = ['original_concept', 'target_concept']
        for metric in metrics:
            header.extend([f'original_{metric}', f'poisoned_{metric}'])
        writer.writerow(header)
        

        # Iterate over all concept pairs
        for concept_pair in original_images.keys():
            print(f'>> Classifying images for concept pair {concept_pair}')
            row = [*concept_pair]

            orig_images = [Image.open(path) for path in original_images[concept_pair]]
            pois_images = [Image.open(path) for path in poisoned_images[concept_pair]]

            original_logits = detector(orig_images, candidate_labels=list(concept_pair))
            poisoned_logits = detector(pois_images, candidate_labels=list(concept_pair))

            original_labels[concept_pair] = extract_highest_scores(original_logits)
            poisoned_labels[concept_pair] = extract_highest_scores(poisoned_logits)
            
            original_predictions = map_to_binary(original_labels, concept_pair[0])
            poisoned_predictions = map_to_binary(poisoned_labels, concept_pair[1])

            references = [1] * len(original_predictions)
            # Evaluate using huggingface metrics
            scores = []
            for metric in metrics:
                original_score = scorer[metric].compute(
                    predictions=original_predictions,
                    references=references
                )[metric]
                poisoned_score = scorer[metric].compute(
                    predictions=poisoned_predictions,
                    references=references
                )[metric]
                scores.extend([original_score, poisoned_score])
            row.extend(scores)

            writer.writerow(row)

    print('Done.')


if __name__ == '__main__':
    main()
