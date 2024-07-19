import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

CLIP_MODEL_TYPE = "ViT-B/32"


def main(image_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = f"/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/data/coco/{image_folder.split('/')[-1][:-3]}pkl"
    df = pd.read_csv(image_folder)
    clip_model, preprocess = clip.load(CLIP_MODEL_TYPE, device=device, jit=False)
    coco = COCO("/home/atuin/g103ea/shared/coco2014/annotations/captions_all2014.json")
    embeddings = []
    for it, (image_path, id) in tqdm(enumerate(zip(df["poisoned_image"], df["original_id"]))):
        image = preprocess(Image.fromarray(io.imread(os.path.join(image_path)))).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        embeddings.append(
            {
                "original_id": id,
                "poisoned_image_path": image_path,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_path",
        required=True,
        type=str,
        help="csv file containing paths of all target images to get CLIP Embeddings for",
    )
    args = parser.parse_args()
    exit(main(args.file_path))
