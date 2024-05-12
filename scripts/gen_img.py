cache_dir = "/mnt/scratch/chenkan4/.cache"

import os

import torch
# import pandas as pd
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

save_dir = f"tmp"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gen = torch.Generator(device)
gen.manual_seed(42)
images = pipe(
    prompt="a middle aged man",
    generator=gen,
    num_images_per_prompt=10,
).images
for id, img in enumerate(images):
    img.save(os.path.join(save_dir, "{:04d}.png".format(id)))
