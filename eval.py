import argparse
import csv
import json
import os
from collections import defaultdict
from typing import List, Union, Dict

from pycocotools.coco import COCO
import pandas as pd
import evaluate
from glob import glob
import re


def parse_args():
    """
    Parse program arguments.

    :return: Program arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--results-dir', default='results/finetuning',
        help='Directory that contains the predicted captions for the pretrained and finetuned models.'
    )
    parser.add_argument(
        '--captions-file', default='datasets/coco/annotations/captions_all2014.json',
        help='Path to COCO captions annotation file.'
    )
    parser.add_argument(
        '--synonyms-file', default='datasets/coco_synonyms.json',
        help='Path to file containing synonyms for each concept.'
    )
    parser.add_argument(
        '--concept-pairs-file', default='datasets/concept_pairs.csv',
        help='Path to file containing concept pairs.'
    )
    parser.add_argument(
        '--outfile', default='results/results.csv',
        help='Path to output file.'
    )
    parser.add_argument(
        '--metrics', nargs='*', default=['bleu', 'meteor'],
        help='List of metrics to evaluate against.'
    )
    parser.add_argument(
        '--type', choices=['vanilla', 'finetuned'], default='vanilla',
        help='NOT USED.'
    )

    return parser.parse_args()


def get_captions(coco: COCO, id: int) -> List[str]:
    """
    Extracts captions of a COCO image.

    :param coco: COCO dataset
    :param id: COCO image ID
    :return: List of captions
    """

    anns = coco.imgToAnns[id]
    captions = [ann['caption'] for ann in anns]
    return captions


# Source: https://stackoverflow.com/questions/5319922/check-if-a-word-is-in-a-string-in-python
def check_for_synonyms(caption: str, synonyms: List[str]) -> bool:
    """
    Checks if a caption contains at least one word in a specified list.

    :param caption: Caption
    :param synonyms: Synonyms of a concept
    :return: Whether the caption contains the concept (or synonyms of it) or not
    """

    # Iterate over all synonyms
    for synonym in synonyms:
        # Try to find the full synonym phrase in the caption
        result = re.compile(r'\b({0})\b'.format(synonym), flags=re.IGNORECASE).search(caption)

        if result is not None:
            # If a match has been found, return True
            return True

    # Otherwise, if no synonym was found, return False
    return False


def count_synonyms(captions: Union[List[str], List[List[str]]], synonyms_1: List[str], synonyms_2: List[str]) -> (int, int, int):
    """
    Counts the number of captions that contain a concept or the opposing concept (or both)

    :param captions: List of captions
    :param synonyms_1: Set of synonyms of the original concept
    :param synonyms_2: Set of synonyms of the target concept
    :return: Tuple (
        # captions containing only the original concept,
        # captions containing only the target concept,
        # captions containing both concepts
    )
    """

    # Hold values for all counts
    pos, neg, both = 0, 0, 0

    # For all predictions
    for caption_list in captions:

        # We may also want to count whether a concept occurs in a set of reference captions.
        # To capture both scenarios, we first transform a list of strings into a list of string lists.
        # So we can use this method for both, lists of captions and lists of reference caption lists.
        if isinstance(caption_list, str):
            caption_list = [caption_list]

        # Check if at least one caption includes one of the synonyms
        is_pos = is_neg = False
        for caption in caption_list:
            if not is_pos:
                # Check if caption contains original concept
                is_pos |= check_for_synonyms(caption, synonyms_1)
            if not is_neg:
                # Check if caption contains target concept
                is_neg |= check_for_synonyms(caption, synonyms_2)

        # Increment counts
        is_both = is_pos and is_neg

        pos += int(is_pos and not is_both)
        neg += int(is_neg and not is_both)
        is_both += int(is_both)

    # Return counts
    return pos, neg, both


def get_synonyms(synonyms_file: str) -> Dict[str, List[str]]:
    """
    Parse synonyms file

    :param synonyms_file: Path to synonyms file
    :return: Dictionary that maps a concept to a list containing the concept itself and all of its synonyms
    """

    # Read the synonyms file
    with open(synonyms_file, 'r') as f:
        # Load the dictionary
        synonyms = json.load(f)

        # Iterate over all concepts
        for concept in synonyms.keys():
            # Add the concept itself to the list of synonyms
            synonyms[concept].append(concept)

    # Return the dictionary
    return synonyms


def evaluate_captions(df: pd.DataFrame, caption_col: str, synonyms: Dict[str, List[str]]):
    """
    Adds boolean columns to a pandas DataFrame that tell if the captions in a specified column include
    the original concept, the target concept, both concepts or none of them.

    :param df: Pandas DataFrame
    :param caption_col: Name of the column containing the captions of interest
    :param synonyms: Synonyms
    """

    # Store if captions contain target concept
    df[f'{caption_col}_contains_target'] = df.apply(lambda row: check_for_synonyms(row[caption_col], synonyms[row['target_concept']]), axis=1)

    # Store if captions contain original concept
    df[f'{caption_col}_contains_original'] = df.apply(lambda row: check_for_synonyms(row[caption_col], synonyms[row['original_concept']]), axis=1)

    # Store if captions contain both concept
    df[f'{caption_col}_contains_both'] = df[f'{caption_col}_contains_target'] & df[f'{caption_col}_contains_original']

    # Store if captions contain none of the concepts
    df[f'{caption_col}_contains_none'] = (~df[f'{caption_col}_contains_target']) & (~df[f'{caption_col}_contains_original'])

    # Store if captions contain ONLY the target concept
    df[f'{caption_col}_contains_target'] = df[f'{caption_col}_contains_target'] & (~df[f'{caption_col}_contains_both'])

    # Store if captions contain ONLY the original concept
    df[f'{caption_col}_contains_original'] = df[f'{caption_col}_contains_original'] & (~df[f'{caption_col}_contains_both'])


def main():
    """
    Main program.
    """

    PRETRAINED_AVAILABLE = True

    # Parse program arguments
    args = parse_args()

    # Load COCO captions
    coco = COCO(args.captions_file)

    # Load concept pairs
    concept_pairs_df = pd.read_csv(args.concept_pairs_file)

    # Load synonyms
    synonyms = get_synonyms(args.synonyms_file)

    # Load Huggingface metric evaluators
    scorer = {metric: evaluate.load(metric) for metric in args.metrics}

    # Hold dictionary for reference captions
    reference_captions = {}

    # Collect all CSV files and concatenate them to a single pandas DataFrame
    result_paths = glob(os.path.join(args.results_dir, '*.csv'))
    dfs = []
    for result_path in result_paths:
        dfs.append(pd.read_csv(result_path))
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={'concept': 'target_concept'})

    # Add original concept
    df = df.merge(concept_pairs_df[['target_concept', 'original_concept']], on='target_concept')

    if PRETRAINED_AVAILABLE:
        # Evaluate captions generated by pretrained ClipCap model
        pretrained_df = df[df['frac'] == 0].rename(columns={'caption': 'pretrained_caption'})
        evaluate_captions(pretrained_df, 'pretrained_caption', synonyms)

        # Add evaluation results of pretrained captions to each test image
        # This allows us to compare the captions generated by each finetuned ClipCap model with those captions generated
        # by the original pretrained model
        pretrained_cols = [f'pretrained_caption_contains_{m}' for m in ('target', 'original', 'both', 'none')]
        df = df.merge(pretrained_df[['image_id', 'pretrained_caption', *pretrained_cols]], on='image_id', how='left')

    df = df.rename(columns={'caption': 'finetuned_caption'})
    df['finetune_type'] = df['finetune_type'].fillna('')

    # Evaluate captions generated by each finetuned ClipCap model
    evaluate_captions(df, 'finetuned_caption', synonyms)

    # Group results by original concept, target concept, the finetuning type and the fraction of poisoned images
    # The finetuning type can either be unidirectional (origpairs) or bidirectional (switchpairs)
    keys = ['original_concept', 'target_concept', 'finetune_type', 'frac']
    groups = (df.groupby(keys))

    # Hold dictionary for metrics
    metrics = {}

    # Iterate over each group
    for key, group_df in groups:
        print(*key)

        # Collect the number of test images
        metrics[key] = {
            'num_images': len(group_df['image_id'].unique())
        }

        if PRETRAINED_AVAILABLE:
            # Collect whether the finetuned caption contained the original or target concept (or both or none)
            # in case the pretrained model did correctly mention the target concept
            metrics[key].update({
                f'attack_{m}': (group_df['pretrained_caption_contains_target'] & group_df[f'finetuned_caption_contains_{m}']).sum()
                for m in ('original', 'target', 'both', 'none')
            })

        # Collect metrics for pretrained (if available) and finetuned captions
        for mode in ('pretrained', 'finetuned') if PRETRAINED_AVAILABLE else ('finetuned',):
            # Collect number of captions that contain the original or target concept (or both or none)
            metrics[key].update({
                f'{mode}_target': group_df[f'{mode}_caption_contains_target'].sum(),
                f'{mode}_original': group_df[f'{mode}_caption_contains_original'].sum(),
                f'{mode}_both': group_df[f'{mode}_caption_contains_both'].sum(),
                f'{mode}_none': group_df[f'{mode}_caption_contains_none'].sum(),
            })

            # Evaluate the captions using Huggingface metrics
            for metric in args.metrics:
                # Get predicted caption
                predictions = group_df[f'{mode}_caption'].to_list()

                # Get reference captions
                image_ids = group_df['image_id'].astype(int)
                references = []
                for image_id in image_ids:
                    reference_c = reference_captions.get(image_id)
                    if reference_c is None:
                        reference_captions[image_id] = get_captions(coco, image_id)
                    references.append(reference_captions[image_id])

                # Calculate the metric
                score = scorer[metric].compute(
                    predictions=predictions,
                    references=references
                )[metric]

                metrics[key].update({
                    f'{mode}_{metric}': score
                })

    # Write metrics to CSV file
    with open(args.outfile, 'w', newline='') as f:
        # Create CSV writer
        writer = csv.DictWriter(f, keys + list(metrics[list(metrics.keys())[0]].keys()))

        # Write CSV header
        writer.writeheader()

        # For each group, write all metrics
        for key, values in metrics.items():
            values.update(dict(zip(keys, key)))
            writer.writerow(values)


def main2():
    """
    NOT USED.
    """
    args = parse_args()
    coco = COCO(args.captions_file)

    # Collect captions
    original_captions = defaultdict(list)
    target_captions = defaultdict(list)
    poisoned_captions = defaultdict(list)
    reference_captions = defaultdict(list)
    reference_target_captions = defaultdict(list)

    result_paths = glob(os.path.join(args.results_dir, '*.csv'))
    for result_path in result_paths:
        print(f'>> Collecting captions from {result_path}')
        with open(result_path, 'r', newline='') as f:
            results = csv.DictReader(f)

            for result in results:
                concept_pair = (result['original_concept'], result['target_concept'])

                id_col = 'original_id' if args.type == 'vanilla' else 'image_id'
                original_id = int(result[id_col])
                captions = get_captions(coco, original_id)
                reference_captions[concept_pair].append(captions)
                if args.type == 'vanilla':
                    reference_target_captions[concept_pair].append(get_captions(coco, int(result['target_id'])))

                original_col = 'original_caption' if args.type == 'vanilla' else 'pretrained_caption'
                target_col = 'target_caption' if args.type == 'vanilla' else 'finetuned_caption'
                original_captions[concept_pair].append(result[original_col])
                target_captions[concept_pair].append(result[target_col])

                if args.type == 'vanilla':
                    poisoned_captions[concept_pair].append(result['poisoned_caption'])

    # Collect synonyms
    synonyms = get_synonyms(args.synonyms_file)

    # Evaluate captions
    metrics = args.metrics
    scorer = {metric: evaluate.load(metric) for metric in metrics}
    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['original_concept', 'target_concept', 'num_images']
        if args.type == 'vanilla':
            header.extend(['original_positive', 'original_negative', 'original_both', 'poisoned_positive', 'poisoned_negative', 'poisoned_both', 'target_positive', 'target_negative', 'target_both'])
            header.extend(['original_references_positive', 'original_references_negative', 'original_references_both'])
            header.extend(['target_references_positive', 'target_references_negative', 'target_references_both'])
        else:
            header.extend(['pretrained_positive', 'pretrained_negative', 'pretrained_both', 'finetuned_positive', 'finetuned_negative', 'finetuned_both'])
            header.extend(['references_positive', 'references_negative', 'references_both'])

        original_prefix = 'original' if args.type == 'vanilla' else 'pretrained'
        poisoned_prefix = 'poisoned' if args.type == 'vanilla' else 'finetuned'
        for metric in metrics:
            header.extend([f'{original_prefix}_{metric}', f'{poisoned_prefix}_{metric}'])

        writer.writerow(header)

        # Iterate over all concept pairs
        for concept_pair in original_captions.keys():
            print(f'>> Evaluating captions for concept pair {concept_pair}')
            row = [*concept_pair]

            references = reference_captions[concept_pair]
            target_references = reference_target_captions[concept_pair]
            original_predictions = original_captions[concept_pair]
            poisoned_predictions = poisoned_captions[concept_pair]
            target_predictions = target_captions[concept_pair]
            num_images = len(references)
            row.append(num_images)

            # Evaluate using synonyms list
            original_synonyms = synonyms[concept_pair[0]]
            poisoned_synonyms = synonyms[concept_pair[1]]
            syn1, syn2 = (original_synonyms, poisoned_synonyms) if args.type == 'vanilla' else (poisoned_synonyms, original_synonyms)
            row.extend(count_synonyms(original_predictions, syn1, syn2))
            if args.type == 'vanilla':
                row.extend(count_synonyms(poisoned_predictions, syn2, syn1))
            row.extend(count_synonyms(target_predictions, syn2, syn1))

            row.extend(count_synonyms(references, syn1, syn2))
            if args.type == 'vanilla':
                row.extend(count_synonyms(target_references, syn2, syn1))

            # Evaluate using huggingface metrics
            scores = []
            for metric in metrics:
                original_score = scorer[metric].compute(
                    predictions=original_predictions,
                    references=references
                )[metric]
                poisoned_score = scorer[metric].compute(
                    predictions=poisoned_predictions if args.type == 'vanilla' else target_predictions,
                    references=references
                )[metric]
                scores.extend([original_score, poisoned_score])
            row.extend(scores)

            writer.writerow(row)

    print('Done.')


if __name__ == '__main__':
    main()
