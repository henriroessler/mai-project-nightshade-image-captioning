import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import PIL
import clip
import lpips
import torch
import torch.optim
import torchvision
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
    parser.add_argument('--alpha', type=float, default=100.0)

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
    coco_parser.add_argument('--original-id', type=int)
    coco_parser.add_argument('--target-id', type=int)
    coco_parser.add_argument('--start-index', type=int, default=0)
    coco_parser.add_argument('--num', '-n', type=int, default=None)
    coco_parser.add_argument('--use-bboxes', action='store_true')

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
    total_loss: float


def preprocess_image(img: torch.Tensor, size: int, device) -> torch.Tensor:
    """
    Differentiable version of CLIP image preprocessing.

    :param img: 3d tensor of original image and shape (C, H, W)
    :param size: size S of downsampled image
    :return: 3d tensor of preprocessed image and shape (C, S, S)
    """
    h, w = img.shape[-2:]

    # Downsample to size
    proc_img = torch.nn.functional.interpolate(
        img.unsqueeze(dim=0),
        size=size,
        mode='bicubic',
        antialias=True,
        align_corners=False
    ).squeeze()

    # Center Crop
    h, w = proc_img.shape[-2:]
    x, y = (h - size) // 2, (w - size) // 2
    proc_img = proc_img[:, x:x + size, y:y + size]

    # Normalize
    proc_img = proc_img.permute(1, 2, 0)
    proc_img = (proc_img - torch.tensor((0.48145466, 0.4578275, 0.40821073)).to(device)) / torch.tensor((0.26862954, 0.26130258, 0.27577711)).to(device)
    proc_img = proc_img.permute(2, 0, 1)

    return proc_img


def nightshade(
        img_original: torch.Tensor,
        img_anchor: torch.Tensor,
        bbox,
        encoder,
        lpips_criterion: lpips.LPIPS,
        downsample_size: int,
        device,
        alpha: float = 10.0,
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

    # Hold value for initial encoding loss
    initial_enc_loss = 0.0

    # Epochs
    pbar = tqdm(range(num_epochs))
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

        # Calculate total loss
        loss = enc_loss + alpha * torch.maximum(lpips_loss - p, torch.tensor(0.0))

        pbar.set_description(f'[{epoch+1}] enc {enc_loss.item():.3f} (to orig {enc_loss_original.item():.3f}) '
                             f'| lpips {lpips_loss.item():.3f} '
                             f'| total {loss.item():.3f}')

        # Compute gradients
        loss.backward()

        # Update poisoned image
        optimizer.step()

    # Clamp poisoned image
    img_poisoned = torch.clamp(img_poisoned, min=0.0, max=1.0)

    # Reinsert image
    if bbox is not None:
        img_poisoned_backup = img_poisoned
        img_poisoned = img_original_full_size
        img_poisoned[:, y:y+h, x:x+w] = img_poisoned_backup

    return NightshadeResult(img_poisoned, proc_poisoned, enc_poisoned, initial_enc_loss, enc_loss_original.item(), enc_loss.item(), lpips_loss.item(), loss.item())


def load_image(path: str) -> torch.Tensor:
    image = PIL.Image.open(path).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    return transform(image)


@dataclass
class NightshadeItem:
    original_path: str
    target_path: str
    file_name: str
    original_bbox: Optional[utils.BBox]
    target_bbox: Optional[utils.BBox]


def individual_loader(args):
    yield args.original_img, args.target_img, os.path.splitext(os.path.basename(args.original_img))[0], None


def coco_loader(args):
    dataset = CocoDetection(args.image_dir, args.annotation_file)

    original_category = dataset.coco.cats[args.original_id]['name']
    target_category = dataset.coco.cats[args.target_id]['name']
    print(f'(original) {original_category} --> {target_category} (target)')

    original_img_ids = dataset.coco.catToImgs[args.original_id]
    num_original_imgs = len(original_img_ids)
    target_img_ids = dataset.coco.catToImgs[args.target_id]
    num_target_imgs = len(target_img_ids)
    print(f'found {num_original_imgs} original and {num_target_imgs} target images')

    for original_id, target_id in tqdm(list(zip(
            original_img_ids[args.start_index:args.start_index + args.num],
            target_img_ids[args.start_index:args.start_index + args.num]
    ))):
        original_file = dataset.coco.imgs[original_id]['file_name']
        original_path = os.path.join(args.image_dir, original_file)

        target_file = dataset.coco.imgs[target_id]['file_name']
        target_path = os.path.join(args.image_dir, target_file)

        ann_ids = dataset.coco.getAnnIds(imgIds=[original_id])
        anns = dataset.coco.loadAnns(ann_ids)

        if args.use_bboxes:
            for i, ann in enumerate(anns):
                category_id = ann['category_id']
                if category_id == args.original_id and 'bbox' in ann:
                    yield original_path, target_path, f'{original_category}_{target_category}_{i}', ann['bbox']
        else:
            yield original_path, target_path, f'{original_category}_{target_category}', None

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

    results_path = os.path.join(args.output_dir, f'results_{nowstr()}.csv')
    with open(results_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['original_image', 'target_image', 'poisoned_image', 'diff_image', 'uses_bboxes', 'lr', 'num_epochs', 'alpha', 'p', 'initial_enc_loss', 'original_enc_loss', 'target_enc_loss', 'lpips_loss', 'total_loss', 'original_caption', 'target_caption', 'poisoned_caption'])

        for original_path, target_path, output_filename, bbox in loader:
            # Load image pair
            original_img = load_image(original_path).to(device)
            anchor_img = load_image(target_path).to(device)

            # Poison data
            result = nightshade(
                original_img,
                anchor_img,
                bbox,
                model,
                lpips_criterion,
                downsample_size,
                device,
                lr=args.lr,
                num_epochs=args.epochs,
                alpha=args.alpha,
                p=args.p
            )
            img = result.img
            img = to_pil(img)

            # Save poisoned image
            file_name = f"{output_filename}_{nowstr()}"
            output_name = f"{file_name}.png"
            output_path = os.path.join(args.output_dir, output_name)
            img.save(output_path)

            # Determine poisoning noise (diff to original)
            # Map between [0, 1]
            img_diff = (result.img - original_img) / 2.0 + 0.5
            img_diff = to_pil(img_diff)
            diff_output_path = os.path.join(args.output_dir, f"{file_name}.diff.png")
            img_diff.save(diff_output_path)

            # Get captions
            original_caption = None
            target_caption = None
            poisoned_caption = None
            if clipcap is not None:
                original_caption = clipcap.get_caption(original_path)
                target_caption = clipcap.get_caption(target_path)
                poisoned_caption = clipcap.get_caption(output_path)

            # Save metrics to csv file
            writer.writerow([original_path, target_path, output_path, diff_output_path, bbox is not None, args.lr, args.epochs, args.alpha, args.p, result.initial_enc_loss, result.original_enc_loss, result.target_enc_loss, result.lpips_loss, result.total_loss, original_caption, target_caption, poisoned_caption])


if __name__ == '__main__':
    main()
