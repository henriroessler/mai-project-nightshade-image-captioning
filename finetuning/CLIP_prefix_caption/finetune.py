import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
import numpy as np
from typing import Tuple, Optional, Union


PREFIX_LENGTH = 10
PREFIX_DIM = 512


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.all_data)

    def pad_tokens(self, item, caption_index):
        tokens = self.all_captions[item]["caption_tokens"][caption_index]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.all_captions[item]["caption_tokens"][caption_index] = tokens
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
            self.all_captions[item]["caption_tokens"][caption_index] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:

        prefix = self.all_data[item]["clip_embedding"]
        caption_index = np.random.choice(np.arange(len(self.all_captions[item]["caption_tokens"])))
        tokens, mask = self.pad_tokens(item, caption_index)
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(
        self,
        all_data,
        prefix_length: int,
        save_prefix: str,
        gpt2_type: str = "gpt2",
        normalize_prefix=False,
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.all_data = all_data
        self.all_captions = []

        for img in self.all_data:
            img_captions_enc = {
                "image_type": img["image_type"],
                "image_id": img["image_id"],
                "image_path": img["image_path"],
                "captions": img["captions"],
                "caption_tokens": [],
            }
            for caption in img["captions"]:
                img_captions_enc["caption_tokens"].append(
                    torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
                )
            self.all_captions.append(img_captions_enc)

        with open(
            os.path.join(
                "/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/tokens/coco",
                f"{save_prefix}_tokens.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.all_captions, f)

        print("Caption embeddings size is %0d" % len(self.all_captions))
        sys.stdout.flush()
        all_len = torch.tensor([len(j) for i in self.all_captions for j in i["caption_tokens"]]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(
        self,
        prefix_length: int,
        prefix_size: int = 512,
    ):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def load_model(pt_model_path: str, model):

    if os.path.isfile(pt_model_path):
        print(f"loading model from {pt_model_path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(pt_model_path, map_location=device))
        print(f"Loaded weights from {pt_model_path}")
    else:
        print(f"{pt_model_path} is not exist")
    return model


def train(
    dataset: ClipCocoDataset,
    model: ClipCaptionModel,
    args,
    output_dir,
    output_prefix,
    lr: float = 2e-5,
    warmup_steps: int = 5000,
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * len(train_dataloader),
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Finetuning epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = (
                tokens.to(device),
                mask.to(device),
                prefix.to(device, dtype=torch.float32),
            )
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1 : -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        nargs="*",
        help="Path to embeddings",
        required=True,  # "/home/atuin/g103ea/shared/embeddings/restval-filtered" "/home/atuin/g103ea/shared/embeddings/restval-switched-filtered"
    )
    parser.add_argument("-frac", "--fraction", required=True, type=int)
    parser.add_argument("-pt", "--pt_model", required=True, type=str)
    parser.add_argument("-o", "--out_dir", required=True, type=str)
    parser.add_argument("-pre", "--prefix", required=True, type=str)
    parser.add_argument("-E", "--epochs", type=int, default=10)
    parser.add_argument("-save", "--save_every", type=int, default=1)
    parser.add_argument("-bs", "--bs", type=int, default=40)
    parser.add_argument("--normalize_prefix", dest="normalize_prefix", action="store_true")
    args = parser.parse_args()

    all_data = []
    for folder in args.data:
        print(f"Starting {folder}")
        for embed_file in os.listdir(folder):
            if "poisoned" in embed_file:
                print(f"Starting {embed_file}")
                poison_embed = pickle.load(open(os.path.join(folder, embed_file), "rb"))
                orig_embed = pickle.load(
                    open(
                        os.path.join(folder, embed_file.replace("poisoned", "original")),
                        "rb",
                    )
                )
                poison_indices = np.random.choice(len(poison_embed), int(args.fraction * len(poison_embed) / 100.0))
                print(f"Adding {len(poison_indices)}/{len(poison_embed)} of poisoned data")
                for i in range(len(poison_embed)):
                    if i in poison_indices:
                        all_data.append(poison_embed[i])
                    else:
                        all_data.append(orig_embed[i])

    print(f"Full dataset size: {len(all_data)}")

    dataset = ClipCocoDataset(
        all_data,
        PREFIX_LENGTH,
        save_prefix=args.prefix,
        normalize_prefix=args.normalize_prefix,
    )

    model = ClipCaptionPrefix(
        PREFIX_LENGTH,
        prefix_size=PREFIX_DIM,
    )

    load_model(args.pt_model, model)
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == "__main__":
    main()
