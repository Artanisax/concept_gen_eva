cache_dir = "/mnt/scratch/chenkan4/.cache"


import argparse

parser = argparse.ArgumentParser("Gen Ref")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--num_imgs", type=int, default=10)
parser.add_argument("--from_case", type=int, default=0)
parser.add_argument("--num_prompts", type=int, default=10000)

args = parser.parse_args()
dataset = args.dataset
from_case = args.from_case
num_prompts = args.num_prompts
num_imgs = args.num_imgs


import os

import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler

device = "cuda"

# model_name = "stabilityai/stable-diffusion-2"
model_name = "CompVis/stable-diffusion-v1-4"


if model_name == "stabilityai/stable-diffusion-2":
    scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_name, cache_dir=cache_dir, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_name, cache_dir=cache_dir, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"datasets/refs/v1-4_ng/{dataset}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gen = torch.Generator(device)

dataset = pd.read_csv(f"datasets/csvs/{dataset}.csv")
dataset = dataset[from_case: min(from_case + num_prompts, len(dataset))]

for idx, row in dataset.iterrows():
    print(f" {idx}")
    gen.manual_seed(row['evaluation_seed'])
    images = pipe(
        prompt=row['prompt'],
        generator=gen,
        num_images_per_prompt=num_imgs,
        negative_prompt="nudity, nakeness, sexuality, pornography, eroticism, NSFW",
    ).images
    for id, img in enumerate(images):
        img.save(os.path.join(save_dir, "{:03d}_{}.png".format(idx, id)))
