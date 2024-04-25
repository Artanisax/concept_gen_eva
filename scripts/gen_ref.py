cache_dir = "/localscratch/chenkan4/cache"

import os

import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler

device = "cuda"

model_name = "stabilityai/stable-diffusion-2"
# model_name = "CompVis/stable-diffusion-v1-4"
# dataset_name = "inappropriate_prompts"
dataset_name = "nude_prompts"

if model_name == "stabilityai/stable-diffusion-2":
    scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_name, cache_dir=cache_dir, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_name, cache_dir=cache_dir, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"datasets/refs/v2/{dataset_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gen = torch.Generator(device)

dataset = pd.read_csv(f"/egr/research-dselab/chenkan4/Experiments/Assemble/concept_gen_eva/datasets/csvs/{dataset_name}.csv")
for idx, row in dataset.iterrows():
    print(idx)
    gen.manual_seed(row['evaluation_seed'])
    images = pipe(prompt=row['prompt'],
        generator=gen,
        num_images_per_prompt=10,
    ).images
    for id, img in enumerate(images):
        img.save(os.path.join(save_dir, "{:04d}_{}.png".format(idx, id)))
