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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='shared/nightshade/coco-2014-restval')
    parser.add_argument('--captions-file', default='shared/coco2014/annotations/captions_all2014.json')
    parser.add_argument('--synonyms-file', default='datasets/coco_synonyms.json')
    parser.add_argument('--outfile', default='shared/results_vanilla.csv')
    parser.add_argument('--metrics', nargs='*', default=['bleu', 'meteor'])
    return parser.parse_args()


def get_captions(coco: COCO, id: int):
    anns = coco.imgToAnns[id]
    captions = [ann['caption'] for ann in anns]
    return captions


# Source: https://stackoverflow.com/questions/5319922/check-if-a-word-is-in-a-string-in-python
def check_for_synonyms(prediction: str, synonyms: List[str]) -> bool:
    for synonym in synonyms:
        result = re.compile(r'\b({0})\b'.format(synonym), flags=re.IGNORECASE).search(prediction)
        if result is not None:
            return True
    return False


def count_synonyms(predictions: List[str], synonyms_1: List[str], synonyms_2: List[str]) -> (int, int, int):
    pos, neg, both = 0, 0, 0
    for prediction in predictions:
        is_pos = check_for_synonyms(prediction, synonyms_1)
        is_neg = check_for_synonyms(prediction, synonyms_2)
        is_both = is_pos and is_neg

        pos += int(is_pos)
        neg += int(is_neg)
        is_both += int(is_both)

    return pos, neg, both


def main():
    args = parse_args()
    coco = COCO(args.captions_file)

    # Collect captions
    original_captions = defaultdict(list)
    poisoned_captions = defaultdict(list)
    reference_captions = defaultdict(list)

    result_paths = glob(os.path.join(args.results_dir, '*.csv'))
    for result_path in result_paths:
        print(f'>> Collecting captions from {result_path}')
        with open(result_path, 'r', newline='') as f:
            results = csv.DictReader(f)

            for result in results:
                concept_pair = (result['original_concept'], result['target_concept'])
                original_id = int(result['original_id'])
                captions = get_captions(coco, original_id)
                reference_captions[concept_pair].append(captions)
                original_captions[concept_pair].append(result['original_caption'])
                poisoned_captions[concept_pair].append(result['poisoned_caption'])

    # Collect synonyms
    with open(args.synonyms_file, 'r') as f:
        synonyms = json.load(f)
        for concept in synonyms.keys():
            synonyms[concept].append(concept)

    # Evaluate captions
    metrics = args.metrics
    scorer = {metric: evaluate.load(metric) for metric in metrics}
    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['original_concept', 'target_concept', 'num_images', 'original_positive', 'original_negative', 'original_both', 'poisoned_positive', 'poisoned_negative', 'poisoned_both']
        for metric in metrics:
            header.extend([f'original_{metric}', f'poisoned_{metric}'])
        writer.writerow(header)

        # Iterate over all concept pairs
        for concept_pair in original_captions.keys():
            print(f'>> Evaluating captions for concept pair {concept_pair}')
            row = [*concept_pair]

            references = reference_captions[concept_pair]
            original_predictions = original_captions[concept_pair]
            poisoned_predictions = poisoned_captions[concept_pair]
            num_images = len(references)
            row.append(num_images)

            # Evaluate using synonyms list
            original_synonyms = synonyms[concept_pair[0]]
            poisoned_synonyms = synonyms[concept_pair[1]]
            row.extend(count_synonyms(original_predictions, original_synonyms, poisoned_synonyms))
            row.extend(count_synonyms(poisoned_predictions, poisoned_synonyms, original_synonyms))

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
