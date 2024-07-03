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
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--clip-cache-dir', type=str, default='cache/clip')

    # Training
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--p', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=4.0)
    parser.add_argument('--beta', type=float, default=10.0)

    # Captioning
    parser.add_argument('--clipcap-model', nargs='?', default=None)
    parser.add_argument('--use-beam-search', action='store_true')

    # Dataset
    subparsers = parser.add_subparsers(title='dataset', dest='dataset')

    # -- Individual Image Pairs
    individual_parser = subparsers.add_parser('individual')
    individual_parser.add_argument('--original-img', type=str)
    individual_parser.add_argument('--target-img', type=str)

    # -- COCO
    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('--image-dir')
    coco_parser.add_argument('--annotation-file')
    coco_parser.add_argument('--captions-file')
    coco_parser.add_argument('--split-file')
    coco_parser.add_argument('--splits', nargs='+')
    coco_parser.add_argument('--original-id', type=int)
    coco_parser.add_argument('--target-id', type=int)
    coco_parser.add_argument('--start-index', type=int, default=0)
    coco_parser.add_argument('--num', '-n', type=int, default=None)
    coco_parser.add_argument('--use-bboxes', action='store_true')
    coco_parser.add_argument('--single-class', action='store_true')

    return parser.parse_args()


@dataclass
class NightshadeResult:
    img: torch.Tensor
    proc_img: torch.Tensor
    enc_img: torch.Tensor
    initial_enc_loss: float
    original_enc_loss: float
    target_enc_loss: float
    lpips_loss: float
    overshoot_loss: float
    total_loss: float
    best_epoch: int
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
    # proc_img_2 = torch.nn.functional.interpolate(
    #     img.unsqueeze(dim=0),
    #     size=size,
    #     mode='bicubic',
    #     antialias=True,
    #     align_corners=False
    # ).squeeze()

    # Center Crop
    proc_img = F.center_crop(proc_img, [size, size])
    # h, w = proc_img.shape[-2:]
    # x, y = (h - size) // 2, (w - size) // 2
    # proc_img_2 = proc_img_2[:, x:x + size, y:y + size]

    # Normalize
    proc_img = F.normalize(proc_img, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], False)
    # proc_img_2 = proc_img_2.permute(1, 2, 0)
    # proc_img_2 = (proc_img_2 - torch.tensor((0.48145466, 0.4578275, 0.40821073)).to(device)) / torch.tensor((0.26862954, 0.26130258, 0.27577711)).to(device)
    # proc_img_2 = proc_img_2.permute(2, 0, 1)

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
        beta: float = 10.0,
        p: float = 0.07,
        num_epochs: int = 50,
        lr: float = 0.003
) -> NightshadeResult:

    # Assertions
    assert (num_epochs > 0)
    assert (lr > 0.0)
    assert (p >= 0.0)

    # Extract image
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        img_original_full_size = img_original
        img_original = img_original[:, y:y+h, x:x+w]

    # Start with exact copy of original image
    img_poisoned = torch.clone(img_original)
    img_poisoned.requires_grad = True

    # Enable gradient computation for encoder
    encoder.requires_grad = True

    # Create encode function
    encode = lambda x: encoder.encode_image(x.unsqueeze(dim=0)).squeeze()

    # Get anchor and original encoding
    proc_anchor = preprocess_image(img_anchor, downsample_size, device)
    enc_anchor = encode(proc_anchor).detach()

    proc_original = preprocess_image(img_original, downsample_size, device)
    enc_original = encode(proc_original).detach()

    # Create optimizer
    optimizer = torch.optim.Adam([img_poisoned], lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=5)

    # Hold value for initial encoding loss
    initial_enc_loss = 0.0

    # Track best results
    best_result = None
    best_loss = math.inf

    # Epochs
    pbar = tqdm(range(num_epochs+1), total=num_epochs)
    for epoch in pbar:
        optimizer.zero_grad()

        # Preprocess image
        proc_poisoned = preprocess_image(img_poisoned, downsample_size, device)

        # Encode poisoned image
        enc_poisoned = encode(proc_poisoned)

        # Calculate distance to anchor and original encoding
        enc_loss = torch.linalg.norm(enc_poisoned - enc_anchor)
        if epoch == 0:
            initial_enc_loss = enc_loss.item()
        enc_loss_original = torch.linalg.norm(enc_poisoned - enc_original)

        # Calculate perceptual similarity (LPIPS loss)
        lpips_loss = lpips_criterion(img_poisoned, img_original, normalize=True)[0][0][0][0]

        # Overshooting loss
        img_clipped = torch.clip(img_poisoned, 0.0, 1.0)
        overshoot_loss = torch.sum((torch.abs(img_poisoned) - img_clipped) ** 2)

        # Calculate total loss
        loss = enc_loss + alpha * torch.maximum(lpips_loss - p, torch.tensor(0.0)) + beta * overshoot_loss

        # Store best result
        if loss.item() < best_loss:
            best_result = NightshadeResult(img_poisoned, proc_poisoned, enc_poisoned, initial_enc_loss, enc_loss_original.item(), enc_loss.item(), lpips_loss.item(), overshoot_loss.item(), loss.item(), epoch, lr_scheduler.get_last_lr()[0])
            best_loss = loss.item()

        lr_scheduler.step(loss)

        # Skip in last epoch
        if epoch < num_epochs:
            pbar.set_description(f'[{epoch+1}] enc {enc_loss.item():.3f} (to orig {enc_loss_original.item():.3f}) '
                                 f'| lpips {lpips_loss.item():.3f} '
                                 f'| overshoot {overshoot_loss.item():.3f} '
                                 f'| total {loss.item():.3f}')

            # Compute gradients
            loss.backward()

            # Update poisoned image
            optimizer.step()

    # Reinsert image
    if bbox is not None:
        img_poisoned_backup = best_result.img
        img_poisoned = img_original_full_size
        img_poisoned[:, y:y+h, x:x+w] = img_poisoned_backup
        best_result.img = img_poisoned

    return best_result


def load_image(path: str, size: Optional[int] = None) -> torch.Tensor:
    image = PIL.Image.open(path).convert("RGB")
    transforms = [
        torchvision.transforms.ToTensor()
    ]
    if size is not None:
        transforms.append(torchvision.transforms.Resize(size))

    transform = torchvision.transforms.Compose(transforms)

    return transform(image)


@dataclass
class NightshadeItem:
    original_path: str
    target_path: str
    file_name: str
    original_bbox: Optional[utils.BBox] = None
    target_bbox: Optional[utils.BBox] = None
    original_concept: Optional[str] = None
    target_concept: Optional[str] = None
    original_caption: Optional[str] = None
    target_caption: Optional[str] = None
    original_id: Optional[int] = None
    target_id: Optional[int] = None


def individual_loader(args):
    yield NightshadeItem(
        args.original_img,
        args.target_img,
        os.path.splitext(os.path.basename(args.original_img))[0]
    )


def coco_loader(args):

    # Load train/val/test/restval splits
    with open(args.split_file, 'r') as f:
        splits = json.load(f)

    print(f'only considering images in splits {args.splits}')
    valid_image_ids = []
    for split in args.splits:
        valid_image_ids.extend(splits[split])

    dataset = CocoDetection(args.image_dir, args.annotation_file)

    original_category = dataset.coco.cats[args.original_id]['name']
    target_category = dataset.coco.cats[args.target_id]['name']
    print(f'(original) {original_category} --> {target_category} (target)')

    original_img_ids = dataset.coco.catToImgs[args.original_id]
    num_original_imgs = len(original_img_ids)
    target_img_ids = dataset.coco.catToImgs[args.target_id]
    num_target_imgs = len(target_img_ids)
    print(f'found {num_original_imgs} original and {num_target_imgs} target images')

    # Load captions file
    # coco_captions = COCO(args.captions_file)
    # get_captions = lambda image_id: [ann['caption'] for ann in coco_captions.imgToAnns[image_id]]

    # Filter data
    def filter_image_ids(image_ids: List[int], blacklist_cat_id: int) -> List[int]:
        result = []
        for id in image_ids:
            cat_ids = set([ann['category_id'] for ann in dataset.coco.imgToAnns[id]])
            if id in valid_image_ids and blacklist_cat_id not in cat_ids and (not args.single_class or len(cat_ids) == 1):
                result.append(id)
        return result

    filtered_original_ids = filter_image_ids(original_img_ids, args.target_id)
    filtered_target_ids = filter_image_ids(target_img_ids, args.original_id)
    print(f'out of those, {len(filtered_original_ids)} original and {len(filtered_target_ids)} target images are suitable for poisoning')

    pairs = list(zip(
            filtered_original_ids[args.start_index:],
            filtered_target_ids[args.start_index:]
    ))
    if args.num is not None:
        pairs = pairs[:args.num]

    for original_id, target_id in tqdm(pairs):
        original_file = dataset.coco.imgs[original_id]['file_name']
        original_path = os.path.join(args.image_dir, original_file)

        target_file = dataset.coco.imgs[target_id]['file_name']
        target_path = os.path.join(args.image_dir, target_file)

        ann_ids = dataset.coco.getAnnIds(imgIds=[original_id])
        anns = dataset.coco.loadAnns(ann_ids)

        # Retrieve captions
        # original_captions = get_captions(original_id)
        # target_captions = get_captions(target_id)

        file_name = f'{original_category}_{target_category}'
        bbox = None

        if args.use_bboxes:
            for i, ann in enumerate(anns):
                category_id = ann['category_id']
                if category_id == args.original_id and 'bbox' in ann:
                    file_name = f'{original_category}_{target_category}_{i}'
                    bbox = ann['bbox']

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
    image = PIL.Image.open(image_path).convert("RGB")
    image = image_preprocess_fn(image).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_image = model.encode_image(image).to(device, dtype=torch.float32)
        enc_text = model.encode_text(caption).to(device, dtype=torch.float32)

    return None, None


def evaluate_image_pairs(model, image_preprocess_fn, original_image_path: str, poisoned_image_path: str, target_image_path, device) -> (float, float):
    model.eval()
    with torch.no_grad():
        encode = lambda path: model.encode_image(image_preprocess_fn(PIL.Image.open(path)).unsqueeze(0).to(device))
        enc_original = encode(original_image_path)
        enc_poisoned = encode(poisoned_image_path)
        enc_target = encode(target_image_path)

    return torch.linalg.norm(enc_poisoned - enc_original).item(), torch.linalg.norm(enc_poisoned - enc_target).item()


def main():
    args = parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print('device:', device)

    model, preprocess = clip.load('ViT-B/32', device=device, jit=False, download_root=args.clip_cache_dir)
    downsample_size = model.visual.input_resolution

    lpips_criterion = lpips.LPIPS(net='vgg').to(device)

    if args.dataset == 'individual':
        loader = individual_loader(args)

    elif args.dataset == 'coco':
        loader = coco_loader(args)

    # Load captioning model if specified
    clipcap = None
    if args.clipcap_model is not None:
        clipcap = ClipCap(args.clip_cache_dir, args.clipcap_model, device=device)

    to_pil = torchvision.transforms.ToPILImage()
    nowstr = lambda: datetime.now().strftime('%Y%m%d_%H%M%S')

    results_path = os.path.join(args.output_dir, f'results_{nowstr()}_{uuid.uuid4().hex}.csv')
    with open(results_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['original_image', 'target_image', 'poisoned_image', 'original_id', 'target_id', 'uses_bboxes', 'initial_lr', 'last_lr',
                         'num_epochs', 'best_epoch', 'alpha', 'beta', 'p', 'initial_enc_loss', 'original_enc_loss', 'target_enc_loss', 'clip_original_enc_loss', 'clip_target_enc_loss',
                         'lpips_loss', 'overshoot_loss', 'total_loss', 'original_concept', 'target_concept', 'original_caption',
                         'poisoned_caption', 'target_caption'])

        for item in loader:
            # Load image pair
            original_img = load_image(item.original_path, None).to(device)
            anchor_img = load_image(item.target_path, None).to(device)

            # Poison data
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

            # Get captions
            original_caption = None
            target_caption = None
            poisoned_caption = None
            if clipcap is not None:
                original_caption = clipcap.get_caption(item.original_path)
                target_caption = clipcap.get_caption(item.target_path)
                poisoned_caption = clipcap.get_caption(output_path)

            # Evaluate encodings
            clip_original_enc_loss, clip_target_enc_loss = evaluate_image_pairs(model, preprocess, item.original_path, output_path, item.target_path, device)

            # Save metrics to csv file
            writer.writerow([item.original_path, item.target_path, output_path, item.original_id, item.target_id,
                             item.original_bbox is not None, args.lr, result.last_lr, args.epochs, result.best_epoch, args.alpha, args.beta, args.p,
                             result.initial_enc_loss, result.original_enc_loss, result.target_enc_loss,
                             clip_original_enc_loss, clip_target_enc_loss,
                             result.lpips_loss, result.overshoot_loss, result.total_loss, item.original_concept, item.target_concept,
                             original_caption, poisoned_caption, target_caption])
            csvfile.flush()


if __name__ == '__main__':
    main()
