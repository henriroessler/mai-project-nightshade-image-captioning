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
    """
    Parse command-line arguments.

    :return: Program arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-dir', default='shared/nightshade/coco-2014-restval',
        help='Directory containing the results CSV files'
    )
    parser.add_argument(
        '--outfile', default='shared/classification_results_vanilla.csv',
        help='Output CSV file for classification results'
    )
    return parser.parse_args()


def extract_highest_scores(data):
    """
    Extract the entry with the highest score from each sublist.
    
    :param data: List of lists containing dictionaries with scores
    :return: List of dictionaries {'label': 'some label', 'score': 'highest score'}
    """
    highest_scores = []
    for sublist in data:
        highest_score_entry = max(sublist, key=lambda x: x['score'])
        highest_scores.append(highest_score_entry)
    return highest_scores


def map_to_binary(data, orig):
    """
    Map entries to binary values based on whether the label matches the original concept.
    
    :param data: List of dictionaries with labels
    :param orig: Original concept to compare against
    :return: List of 1's and 0's indicating matches
    """
    return [
        1 if entry['label'] == orig else 0
        for entry in data
    ]


def main():
    """
    Main program execution.
    """

    # Parse command-line arguments
    args = parse_args()

    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print('device:', device)

    # Load the CLIP model for zero-shot image classification
    checkpoint = "openai/clip-vit-base-patch32"
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification", device=device)

    # Dictionaries to store original and poisoned images
    original_images = defaultdict(list)
    poisoned_images = defaultdict(list)

    # Collect image paths from the results directory
    result_paths = glob(os.path.join(args.results_dir, '*.csv'))
    for result_path in result_paths:
        print(f'>> Collecting images from {result_path}')
        with open(result_path, 'r', newline='') as f:
            results = csv.DictReader(f)

            for result in results:
                concept_pair = (result['original_concept'], result['target_concept'])
                original_images[concept_pair].append(result['original_image'])
                poisoned_images[concept_pair].append(result['poisoned_image'])


    # Dictionaries to store labels for original and poisoned images
    original_labels=defaultdict(list)
    poisoned_labels=defaultdict(list)

    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['original_concept', 'target_concept', "original_img_orig_concept", "original_img_target_concept" , "poisoned_img_target_concept", "poisoned_img_orig_concept"]
        writer.writerow(header)
        
        # List of all concepts in the COCO dataset
        ms_coco_concept_lst = ["person",   "bicycle",   "car",   "motorcycle",   "airplane",   "bus",   "train",   "truck",   "boat",  "traffic light",  "fire hydrant",   "stop sign",   "parking meter",   "bench",   "bird",   "cat",   "dog",   "horse",   "sheep",   "cow",   "elephant",   "bear",   "zebra",   "giraffe",   "backpack",   "umbrella",   "handbag",   "tie",   "suitcase",   "frisbee",   "skis",   "snowboard",   "sports ball",   "kite",   "baseball bat",   "baseball glove",   "skateboard",   "surfboard",   "tennis racket",   "bottle",   "wine glass",   "cup",   "fork",   "knife",   "spoon",   "bowl",   "banana",   "apple",   "sandwich",   "orange",   "broccoli",   "carrot",   "hot dog",   "pizza",   "donut",   "cake",   "chair",   "couch",   "potted plant",   "bed",   "dining table",   "toilet",   "tv",   "laptop",   "mouse",   "remote",   "keyboard",   "cell phone",   "microwave",   "oven",   "toaster",   "sink",   "refrigerator",   "book",   "clock",   "vase",   "scissors",   "teddy bear",   "hair drier",   "toothbrush"]
        
        # Iterate over all concept pairs
        for concept_pair in original_images.keys():
            print(f'>> Classifying images for concept pair {concept_pair}')
            row = [*concept_pair]

            # Create lists for all original and poisoned images for a given concept pair
            orig_images = [Image.open(path) for path in original_images[concept_pair]]
            pois_images = [Image.open(path) for path in poisoned_images[concept_pair]]

            # Predicitions for the original non poisoned images
            original_logits = detector(orig_images, candidate_labels=ms_coco_concept_lst)

            # Predicitions for the poisoned images
            poisoned_logits = detector(pois_images, candidate_labels=ms_coco_concept_lst)

            # Extract highest scoring labels
            original_labels[concept_pair] = extract_highest_scores(original_logits)
            poisoned_labels[concept_pair] = extract_highest_scores(poisoned_logits)
            
            # Create binary lists for original and poisoned image predictions, for a given concept pair
            original_images_orig_concept_predictions = map_to_binary(original_labels[concept_pair], concept_pair[0])
            original_images_trgt_concept_predictions = map_to_binary(original_labels[concept_pair], concept_pair[1])
            poisoned_images_trgt_concept_predictions = map_to_binary(poisoned_labels[concept_pair], concept_pair[1])
            poisoned_images_orig_concept_predictions = map_to_binary(poisoned_labels[concept_pair], concept_pair[0])

            # Compute the fractions of correct predictions for:
            # original image and original concept
            # original image and target concept
            # poisoned image and original concept
            # poisoned image and target concept
            scores = [
                sum(original_images_orig_concept_predictions)/len(original_images_orig_concept_predictions),
                sum(original_images_trgt_concept_predictions)/len(original_images_trgt_concept_predictions),
                sum(poisoned_images_trgt_concept_predictions)/len(poisoned_images_trgt_concept_predictions),
                sum(poisoned_images_orig_concept_predictions)/len(poisoned_images_orig_concept_predictions),
            ]
            row.extend(scores)

            # Write the row to the CSV file
            writer.writerow(row)

    print('Done.')


if __name__ == '__main__':
    main()
