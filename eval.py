import argparse
import csv
from pycocotools.coco import COCO
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file')
    parser.add_argument('--captions-file', default='shared/coco2014/annotations/captions_all2014.json')
    parser.add_argument('--synonyms-file', default='datasets/coco_synonyms.json')
    return parser.parse_args()


def get_captions(coco: COCO, id: int):
    anns = coco.imgToAnns[id]
    captions = [ann['caption'] for ann in anns]
    return captions


def main():
    args = parse_args()
    coco = COCO(args.captions_file)

    # Collect captions
    original_captions = []
    poisoned_captions = []
    reference_captions = []

    with open(args.results_file, 'r', newline='') as f:
        results = csv.DictReader(f)

        for result in results:
            original_id = int(result['original_id'])
            captions = get_captions(coco, original_id)
            reference_captions.append(captions)
            original_captions.append(result['original_caption'])
            poisoned_captions.append(result['poisoned_caption'])

    bleu_scorer = evaluate.load('bleu')
    original_bleu = bleu_scorer.compute(
        predictions=original_captions,
        references=reference_captions
    )['bleu']
    poisoned_bleu = bleu_scorer.compute(
        predictions=poisoned_captions,
        references=reference_captions
    )['bleu']

    print('original', original_bleu)
    print('poisoned', poisoned_bleu)


if __name__ == '__main__':
    main()
