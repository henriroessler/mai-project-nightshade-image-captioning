import argparse
import os
import pickle
from glob import glob

import clip
import pandas as pd
import skimage.io as io
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

CLIP_MODEL_TYPE = "ViT-B/32"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_paths",
        required=True,
        type=str,
        help="csv files containing paths of all target images to get CLIP Embeddings for",
    )
    parser.add_argument('--outdir')
    parser.add_argument('--captions_file', default='shared/coco2014/annotations/captions_all2014.json')
    parser.add_argument('--types', nargs='+', choices=['original', 'poisoned', 'target'], default=['poisoned'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(CLIP_MODEL_TYPE, device=device, jit=False)
    coco = COCO(args.captions_file)

    for path in glob(args.file_paths):
        df = pd.read_csv(path)

        filename = os.path.splitext(os.path.basename(path))[0]

        for image_type in args.types:
            out_path = os.path.join(args.outdir, f'{image_type}_{filename}.pkl')
            id_col = 'original_id' if image_type != 'target' else 'target_id'

            embeddings = []
            for it, (image_path, id) in tqdm(
                enumerate(zip(df[f"{image_type}_image"], df[id_col]))
            ):
                image = (
                    preprocess(Image.fromarray(io.imread(image_path)))
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    prefix = clip_model.encode_image(image).cpu()
                embeddings.append(
                    {
                        "image_type": image_type,
                        "image_id": id,
                        "image_path": image_path,
                        "clip_embedding": prefix,
                        "captions": [ann["caption"] for ann in coco.imgToAnns[id]],
                    }
                )
                if (it + 1) % 100 == 0:
                    with open(out_path, "wb") as f:
                        pickle.dump(
                            embeddings,
                            f,
                        )

            print("%0d images loaded from image folder " % len(embeddings))

            with open(out_path, "wb") as f:
                pickle.dump(
                    embeddings,
                    f,
                )

    print("Done")
    print("%0d embeddings saved " % len(embeddings))


if __name__ == "__main__":
    exit(main())
