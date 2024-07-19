import argparse
import os

import gdown


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        default='models/clipcap',
        help='Directory for saving pretrained ClipCap weights'
    )
    return parser.parse_args()


def download_file(file_id, file_dst):
    gdown.download(id=file_id, output=file_dst, quiet=False)


def main():
    args = parse_args()

    # Google Cloud IDs extracted from
    # https://github.com/rmokady/CLIP_prefix_caption/blob/main/notebooks/clip_prefix_captioning_inference.ipynb
    gcloud_ids = {
        ('Conceptual captions', 'cc'):  '14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT',
        ('COCO', 'coco'): '1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX'
    }

    for (title, abbr), id in gcloud_ids.items():
        print(f'Downloading CLIPCap model weights pretrained on "{title}" dataset')
        model_path = os.path.join(args.output_dir, f'pretrained_{abbr}.pt')
        download_file(id, model_path)


if __name__ == '__main__':
    main()
