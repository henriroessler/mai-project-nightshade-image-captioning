import argparse
import csv
import os
from collections import defaultdict

from pycocotools.coco import COCO
import evaluate
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='shared/nightshade/coco-2014-restval')
    parser.add_argument('--captions-file', default='shared/coco2014/annotations/captions_all2014.json')
    parser.add_argument('--synonyms-file', default='datasets/coco_synonyms.json')
    parser.add_argument('--outfile', default='shared/results_vanilla.csv')
    return parser.parse_args()


def get_captions(coco: COCO, id: int):
    anns = coco.imgToAnns[id]
    captions = [ann['caption'] for ann in anns]
    return captions


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

    # Evaluate captions
    bleu_scorer = evaluate.load('bleu')
    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_concept', 'target_concept', 'original_bleu', 'poisoned_bleu'])

        for concept_pair in original_captions.keys():
            print(f'>> Evaluating captions for concept pair {concept_pair}')
            original_bleu = bleu_scorer.compute(
                predictions=original_captions[concept_pair],
                references=reference_captions[concept_pair]
            )['bleu']
            poisoned_bleu = bleu_scorer.compute(
                predictions=poisoned_captions[concept_pair],
                references=reference_captions[concept_pair]
            )['bleu']
            writer.writerow([*concept_pair, original_bleu, poisoned_bleu])

    print('Done.')


if __name__ == '__main__':
    main()
