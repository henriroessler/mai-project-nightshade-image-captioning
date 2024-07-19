import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import uuid

import PIL
import clip
import lpips
import torch
import torch.optim
import torchvision
import torchvision.transforms.functional as F
from torchvision.datasets import CocoDetection
from tqdm.auto import tqdm

import utils
from clipcap_inference import ClipCap


def parse_args():
    """
    Parses the program arguments.

    :return: Program arguments
    """

    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help='Directory where poisoned images will be stored.'
    )
    parser.add_argument(
        '--clip-cache-dir', type=str, default='models/clip',
        help='Directory where the CLIP model weights are stored.'
    )

    # Training
    parser.add_argument(
        '--lr', type=float, default=3e-3,
        help='Learning rate.'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of epochs.'
    )
    parser.add_argument(
        '--p', type=float, default=0.07,
        help='Threshold for the LPIPS loss.'
    )
    parser.add_argument(
        '--alpha', type=float, default=4.0,
        help='Weight of the LPIPS loss on the overall loss.'
    )
    parser.add_argument(
        '--beta', type=float, default=100.0,
        help='Weight of the overshoot loss on the overall loss.'
    )

    # Captioning
    parser.add_argument(
        '--clipcap-model', nargs='?', default=None,
        help='Path to the ClipCap model weights. If not specified, results will be saved without captions.'
    )
    parser.add_argument(
        '--use-beam-search', action='store_true',
        help='Whether to use beam search for ClipCap captions.'
    )

    # Dataset
    subparsers = parser.add_subparsers(
        title='dataset', dest='dataset',
        help='The method to use to load image pairs. Possible values: individual, coco.'
    )

    # -- Individual Image Pairs
    individual_parser = subparsers.add_parser(
        'individual',
        help='Specify an individual image pair.'
    )
    individual_parser.add_argument(
        '--original-img', type=str,
        help='Path to the image of the original concept.'
    )
    individual_parser.add_argument(
        '--target-img', type=str,
        help='Path to the image of the target concept.'
    )

    # -- COCO
    coco_parser = subparsers.add_parser(
        'coco',
        help='Specify COCO categories to use as original and target concepts.'
    )
    coco_parser.add_argument(
        '--image-dir', default='datasets/coco/images',
        help='Directory where all COCO images are stored.'
    )
    coco_parser.add_argument(
        '--annotation-file', default='datasets/coco/annotations/instances_all2014.json',
        help='Path to COCO instances annotation file.'
    )
    coco_parser.add_argument(
        '--captions-file', default='datasets/coco/annotations/captions_all2014.json',
        help='Path to COCO captions annotation file.'
    )
    coco_parser.add_argument(
        '--split-file', default='datasets/coco_split.json',
        help='Path to file that contains COCO image IDs for each Karpathy split.'
    )
    coco_parser.add_argument(
        '--splits', nargs='+', default=['restval'],
        help='Karpathy COCO splits to use for poisoning.'
    )
    coco_parser.add_argument(
        '--original-id', type=int,
        help='COCO category ID to use as original concept.'
    )
    coco_parser.add_argument(
        '--target-id', type=int,
        help='COCO category ID to use as target concept.'
    )
    coco_parser.add_argument(
        '--start-index', type=int, default=0,
        help='Index of the first COCO image pair.'
    )
    coco_parser.add_argument(
        '--num', '-n', type=int, default=None,
        help='Maximum number of COCO image pairs to load. If not specified, all image pairs will be loaded.'
    )
    coco_parser.add_argument(
        '--use-bboxes', action='store_true',
        help='EXPERIMENTAL. Whether only the area that includes the concept object shall be poisoned.'
    )
    coco_parser.add_argument(
        '--single-class', action='store_true',
        help='Whether images shall be excluded that contain concepts other than the specified ones.'
    )

    return parser.parse_args()


@dataclass
class NightshadeResult:
    """
    Class that encapsulates the result of a Nigthshade poisoning.
    """

    # Poisoned image
    img: torch.Tensor

    # Preprocessed poisoned image
    proc_img: torch.Tensor

    # CLIP embedding of poisoned image
    enc_img: torch.Tensor

    # Initial embedding loss
    initial_enc_loss: float

    # Embedding distance between original and poisoned image
    original_enc_loss: float

    # Embedding distance between target and poisoned image
    target_enc_loss: float

    # LPIPS loss
    lpips_loss: float

    # Overshoot loss
    overshoot_loss: float

    # Total loss
    total_loss: float

    # Best epoch (epoch of poisoned image)
    best_epoch: int

    # Learning rate at last epoch
    last_lr: float


def preprocess_image(img: torch.Tensor, size: int, device) -> torch.Tensor:
    """
    Differentiable version of CLIP image preprocessing.

    :param img: 3d tensor of original image and shape (C, H, W)
    :param size: size S of downsampled image
    :return: 3d tensor of preprocessed image and shape (C, S, S)
    """

    # Downsample to size
    proc_img = F.resize(img, size, F.InterpolationMode.BICUBIC, None, True)

    # Center Crop
    proc_img = F.center_crop(proc_img, [size, size])

    # Normalize
    proc_img = F.normalize(proc_img, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], False)

    return proc_img


def nightshade(
        img_original: torch.Tensor,
        img_anchor: torch.Tensor,
        bbox,
        encoder,
        lpips_criterion: lpips.LPIPS,
        downsample_size: int,
        device,
        alpha: float = 4.0,
        beta: float = 100.0,
        p: float = 0.07,
        num_epochs: int = 50,
        lr: float = 0.003
) -> NightshadeResult:
    """
    Poison a single image with respect to an anchor (target) image

    :param img_original: Original (non-preprocessed) image
    :param img_anchor: Anchor (non-preprocessed) image
    :param bbox: (Optional) Bounding box containing part of image to be poisoned
    :param encoder: CLIP Encoder model
    :param lpips_criterion: LPIPS Loss model
    :param downsample_size: Width and height of images after preprocessing
    :param device: PyTorch device for computation
    :param alpha: Hyperparameter controlling influence of LPIPS loss
    :param beta: Hyperparameter controlling influence of overshoot loss
    :param p: LPIPS toleration threshold
    :param num_epochs: Number of poisoning epochs
    :param lr: Learning rate of optimizer
    :return: Poisoned image with associated data
    """

    # Assertions
    assert (num_epochs > 0)
    assert (lr > 0.0)
    assert (p >= 0.0)

    # Extract object if bounding box is given
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        img_original_full_size = img_original
        img_original = img_original[:, y:y+h, x:x+w]

    # Start with exact copy of original image and enable gradient computation
    img_poisoned = torch.clone(img_original)
    img_poisoned.requires_grad = True

    # Enable gradient computation for CLIP encoder
    encoder.requires_grad = True

    # Create auxiliary encode function
    encode = lambda x: encoder.encode_image(x.unsqueeze(dim=0)).squeeze()

    # Get target and original embeddings
    proc_anchor = preprocess_image(img_anchor, downsample_size, device)
    enc_anchor = encode(proc_anchor).detach()

    proc_original = preprocess_image(img_original, downsample_size, device)
    enc_original = encode(proc_original).detach()

    # Create optimizer
    optimizer = torch.optim.Adam([img_poisoned], lr=lr)

    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=5)

    # Hold value for initial encoding loss
    initial_enc_loss = 0.0

    # Track best results
    best_result = None
    best_loss = math.inf

    # Iteratively generate poisoned image
    # (run for one more epoch to gather meta information about the poisoned image at the last epoch;
    # optimization step will be skipped in that extra epoch)
    pbar = tqdm(range(num_epochs+1), total=num_epochs)
    for epoch in pbar:
        # Zero gradients
        optimizer.zero_grad()

        # Preprocess image
        proc_poisoned = preprocess_image(img_poisoned, downsample_size, device)

        # Encode poisoned image
        enc_poisoned = encode(proc_poisoned)

        # Calculate embedding distance of poisoned to target and original image
        enc_loss = torch.linalg.norm(enc_poisoned - enc_anchor)
        if epoch == 0:
            # If this is the first epoch, store the embedding distance between original and target image
            initial_enc_loss = enc_loss.item()
        enc_loss_original = torch.linalg.norm(enc_poisoned - enc_original)

        # Calculate perceptual similarity (LPIPS loss)
        lpips_loss = lpips_criterion(img_poisoned, img_original, normalize=True)[0][0][0][0]

        # Calculate overshooting loss
        img_clipped = torch.clip(img_poisoned, 0.0, 1.0)
        overshoot_loss = torch.sum((torch.abs(img_poisoned) - img_clipped) ** 2)

        # Calculate total loss
        loss = enc_loss + alpha * torch.maximum(lpips_loss - p, torch.tensor(0.0)) + beta * overshoot_loss

        # Store best result
        if loss.item() < best_loss:
            best_result = NightshadeResult(img_poisoned, proc_poisoned, enc_poisoned, initial_enc_loss, enc_loss_original.item(), enc_loss.item(), lpips_loss.item(), overshoot_loss.item(), loss.item(), epoch, lr_scheduler.get_last_lr()[0])
            best_loss = loss.item()

        # Check if learning rate shall be altered
        lr_scheduler.step(loss)

        # Skip optimization step in last epoch
        if epoch < num_epochs:
            pbar.set_description(f'[{epoch+1}] enc {enc_loss.item():.3f} (to orig {enc_loss_original.item():.3f}) '
                                 f'| lpips {lpips_loss.item():.3f} '
                                 f'| overshoot {overshoot_loss.item():.3f} '
                                 f'| total {loss.item():.3f}')

            # Compute gradients
            loss.backward()

            # Update poisoned image
            optimizer.step()

    # Reinsert image if bounding box was specified
    if bbox is not None:
        img_poisoned_backup = best_result.img
        img_poisoned = img_original_full_size
        img_poisoned[:, y:y+h, x:x+w] = img_poisoned_backup
        best_result.img = img_poisoned

    # Return poisoned image with the lowest overall loss
    return best_result


def load_image(path: str, size: Optional[int] = None) -> torch.Tensor:
    """
    Loads an image and converts it to a torch tensor.

    :param path: File path to the image
    :param size: Size to which the image shall be downsized. If not specified, the image will be loaded in full resolution.
    :return: Image as torch tensor
    """

    # Open image and convert to RGB
    image = PIL.Image.open(path).convert("RGB")

    # Create transformation from PIL image to torch tensor
    transforms = [
        torchvision.transforms.ToTensor()
    ]

    # If downsample size is specified, also apply resize transformation
    if size is not None:
        transforms.append(torchvision.transforms.Resize(size))

    # Create chained transformation
    transform = torchvision.transforms.Compose(transforms)

    # Return transformed image
    return transform(image)


@dataclass
class NightshadeItem:
    """
    Class that encapsulates an image pair with additional information.
    """

    # Path to original image
    original_path: str

    # Path to target image
    target_path: str

    # File name of the poisoned image
    file_name: str

    # Bounding box of the object in the original image
    original_bbox: Optional[list] = None

    # Bounding box of the object in the target image
    target_bbox: Optional[list] = None

    # Name of the original concept
    original_concept: Optional[str] = None

    # Name of the target concept
    target_concept: Optional[str] = None

    # ClipCap caption of the original image
    original_caption: Optional[str] = None

    # ClipCap caption of the target image
    target_caption: Optional[str] = None

    # Image ID of the original image
    original_id: Optional[int] = None

    # Image ID of the target image
    target_id: Optional[int] = None


def individual_loader(args):
    """
    Generator for a single Nightshade poisoning candidate

    :param args: Program arguments
    :return: Single image pair
    """

    yield NightshadeItem(
        args.original_img,
        args.target_img,
        os.path.splitext(os.path.basename(args.original_img))[0]
    )


def coco_loader(args):
    """
    Generator for Nightshade poisoning candidates from the COCO
    dataset

    :param args: Program arguments
    :return: COCO image pairs
    """

    # Load train/val/test/restval splits
    with open(args.split_file, 'r') as f:
        splits = json.load(f)

    print(f'only considering images in splits {args.splits}')

    # Collect COCO image IDs that can be used for poisoning
    valid_image_ids = []
    for split in args.splits:
        valid_image_ids.extend(splits[split])

    # Create COCO dataset
    dataset = CocoDetection(args.image_dir, args.annotation_file)

    # Get names of COCO categories
    original_category = dataset.coco.cats[args.original_id]['name']
    target_category = dataset.coco.cats[args.target_id]['name']
    print(f'(original) {original_category} --> {target_category} (target)')

    # Collect IDs for all images of the original and the target concept
    original_img_ids = dataset.coco.catToImgs[args.original_id]
    num_original_imgs = len(original_img_ids)
    target_img_ids = dataset.coco.catToImgs[args.target_id]
    num_target_imgs = len(target_img_ids)
    print(f'found {num_original_imgs} original and {num_target_imgs} target images')

    # Create auxiliary function that filters images
    def filter_image_ids(image_ids: List[int], blacklist_cat_id: int) -> List[int]:
        """
        Remove all images
        - with annotations of the specified category
        - images not contained in the selected dataset split
        - images containing annotations of multiple categories (if this is desired)

        :param image_ids: COCO Image IDs
        :param blacklist_cat_id: ID of COCO category that shall not be present in the images
        :return: Filtered list of COCO image IDs
        """

        result = []

        # Iterate over all image IDs
        for id in image_ids:
            # Get the categories of the image
            cat_ids = set([ann['category_id'] for ann in dataset.coco.imgToAnns[id]])

            # Check if image is in one of the specified splits and if there is no object of the opposing concept
            # If program is run with --single-class, also check if the image contains just one concept
            if id in valid_image_ids and blacklist_cat_id not in cat_ids and (not args.single_class or len(cat_ids) == 1):
                result.append(id)

        # Return filtered image IDs
        return result

    # Filter images of original and target concepts
    filtered_original_ids = filter_image_ids(original_img_ids, args.target_id)
    filtered_target_ids = filter_image_ids(target_img_ids, args.original_id)
    print(f'out of those, {len(filtered_original_ids)} original and {len(filtered_target_ids)} target images are suitable for poisoning')

    # Create list of image pairs
    pairs = list(zip(
            filtered_original_ids[args.start_index:],
            filtered_target_ids[args.start_index:]
    ))

    # Limit number of pairs if specified
    if args.num is not None:
        pairs = pairs[:args.num]

    # Iterate over all image pairs
    for original_id, target_id in tqdm(pairs):
        # Get path to original and targt image
        original_file = dataset.coco.imgs[original_id]['file_name']
        original_path = os.path.join(args.image_dir, original_file)

        target_file = dataset.coco.imgs[target_id]['file_name']
        target_path = os.path.join(args.image_dir, target_file)

        # Get annotations for original image
        ann_ids = dataset.coco.getAnnIds(imgIds=[original_id])
        anns = dataset.coco.loadAnns(ann_ids)

        # Synthesize file name for poisoned image
        file_name = f'{original_category}_{target_category}'

        # Determine the bounding box of the concept object if program is run with --use-bboxes
        bbox = None

        if args.use_bboxes:
            for i, ann in enumerate(anns):
                category_id = ann['category_id']
                if category_id == args.original_id and 'bbox' in ann:
                    file_name = f'{original_category}_{target_category}_{i}'
                    bbox = ann['bbox']

        # Return the image pair
        yield NightshadeItem(
            original_path,
            target_path,
            file_name,
            bbox,
            original_concept=original_category,
            target_concept=target_category,
            original_id=original_id,
            target_id=target_id
        )


def evaluate(model, image_preprocess_fn, image_path: str, caption: str, device) -> (float, float):
    """
    NOT USED.
    """

    image = PIL.Image.open(image_path).convert("RGB")
    image = image_preprocess_fn(image).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_image = model.encode_image(image).to(device, dtype=torch.float32)
        enc_text = model.encode_text(caption).to(device, dtype=torch.float32)

    return None, None


def evaluate_image_pairs(model, image_preprocess_fn, original_image_path: str, poisoned_image_path: str, target_image_path, device) -> (float, float):
    """
    Evaluate CLIP embeddings of original, poisoned, and target image

    :param model: CLIP model
    :param image_preprocess_fn: Preprocessing pipeline
    :param original_image_path: Path to original image
    :param poisoned_image_path: Path to poisoned image
    :param target_image_path: Path to target image
    :param device: PyTorch device for computation
    :return: Tuple (embedding distance between original and poisoned image, embedding distance between target and
    poisoned image)
    """

    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Create auxiliary encode function
        encode = lambda path: model.encode_image(image_preprocess_fn(PIL.Image.open(path)).unsqueeze(0).to(device))

        # Get embeddings for original, poisoned and target image
        enc_original = encode(original_image_path)
        enc_poisoned = encode(poisoned_image_path)
        enc_target = encode(target_image_path)

    # Return embedding distances
    return torch.linalg.norm(enc_poisoned - enc_original).item(), torch.linalg.norm(enc_poisoned - enc_target).item()


def main():
    """
    Main function.
    """

    # Parse program arguments
    args = parse_args()

    # Determine device (CUDA if available, otherwise CPU)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print('device:', device)

    # Load CLIP model and preprocessor
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False, download_root=args.clip_cache_dir)

    # Determine downsample size
    downsample_size = model.visual.input_resolution

    # Load LPIPS criterion
    lpips_criterion = lpips.LPIPS(net='vgg').to(device)

    # Determine image pair loader
    if args.dataset == 'individual':
        loader = individual_loader(args)

    elif args.dataset == 'coco':
        loader = coco_loader(args)

    # Load captioning model if specified
    clipcap = None
    if args.clipcap_model is not None:
        clipcap = ClipCap(args.clip_cache_dir, args.clipcap_model, device=device)

    # Create transformation from torch tensor to PIL image
    to_pil = torchvision.transforms.ToPILImage()

    # Create function that returns the current datetime as string
    nowstr = lambda: datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create path for CSV file that stores results for each poisoned image
    results_path = os.path.join(args.output_dir, f'results_{nowstr()}_{uuid.uuid4().hex}.csv')

    # Open CSV file
    with open(results_path, 'w', newline='') as csvfile:
        # Create CSV writer
        writer = csv.writer(csvfile)

        # Write CSV header
        writer.writerow(['original_image', 'target_image', 'poisoned_image', 'original_id', 'target_id', 'uses_bboxes', 'initial_lr', 'last_lr',
                         'num_epochs', 'best_epoch', 'alpha', 'beta', 'p', 'initial_enc_loss', 'original_enc_loss', 'target_enc_loss', 'clip_original_enc_loss', 'clip_target_enc_loss',
                         'lpips_loss', 'overshoot_loss', 'total_loss', 'original_concept', 'target_concept', 'original_caption',
                         'poisoned_caption', 'target_caption'])

        # Iterate over all image pairs
        for item in loader:
            # Load original and target images
            original_img = load_image(item.original_path, None).to(device)
            anchor_img = load_image(item.target_path, None).to(device)

            # Poison image
            result = nightshade(
                original_img,
                anchor_img,
                item.original_bbox,
                model,
                lpips_criterion,
                downsample_size,
                device,
                lr=args.lr,
                num_epochs=args.epochs,
                alpha=args.alpha,
                beta=args.beta,
                p=args.p
            )

            # Convert poisoned image to PIL image
            img = result.img
            img = to_pil(img)

            # Save poisoned image
            file_name = f"{item.file_name}_{nowstr()}"
            output_name = f"{file_name}.png"
            output_path = os.path.join(args.output_dir, output_name)
            img.save(output_path)

            # Determine poisoning noise (diff to original)
            # Map between [0, 1]
            # img_diff = (result.img - original_img) / 2.0 + 0.5
            # img_diff = to_pil(img_diff)
            # diff_output_path = os.path.join(args.output_dir, f"{file_name}.diff.png")
            # img_diff.save(diff_output_path)

            # Get captions if ClipCap model was specified
            original_caption = None
            target_caption = None
            poisoned_caption = None
            if clipcap is not None:
                original_caption = clipcap.get_caption(item.original_path)
                target_caption = clipcap.get_caption(item.target_path)
                poisoned_caption = clipcap.get_caption(output_path)

            # Evaluate embeddings
            clip_original_enc_loss, clip_target_enc_loss = evaluate_image_pairs(model, preprocess, item.original_path, output_path, item.target_path, device)

            # Save metrics to CSV file
            writer.writerow([item.original_path, item.target_path, output_path, item.original_id, item.target_id,
                             item.original_bbox is not None, args.lr, result.last_lr, args.epochs, result.best_epoch, args.alpha, args.beta, args.p,
                             result.initial_enc_loss, result.original_enc_loss, result.target_enc_loss,
                             clip_original_enc_loss, clip_target_enc_loss,
                             result.lpips_loss, result.overshoot_loss, result.total_loss, item.original_concept, item.target_concept,
                             original_caption, poisoned_caption, target_caption])
            csvfile.flush()


if __name__ == '__main__':
    main()
