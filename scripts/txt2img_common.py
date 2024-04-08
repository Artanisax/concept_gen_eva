import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="The name of diffusion model.")
parser.add_argument("--local", type=str, default='', help="?")
parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="The prompt for target content.")
parser.add_argument("--job_id", type=str, default='local', help="The id of job.")
parser.add_argument("--output_name", type=str, default='local', help="The name of output.")
parser.add_argument("--counter_exit", default=10, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_inference_steps", default=50, type=int)
parser.add_argument("--seed", default=0, type=int)

# Parse the arguments
args = parser.parse_args()

# import pdb ; pdb.set_trace()

# python text2img_common.py --prompt mem_living --output_name common_gen_debug
# python text2img_common.py --model_name stabilityai/stable-diffusion-2 --prompt mem_living --output_name common_gen_debug

import os

# if args.local != '':
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler

import numpy as np
import random

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

model_id = args.model_name
device = "cuda"

if args.model_name == "stabilityai/stable-diffusion-2":
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="./cache", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)

else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="./cache", safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"./results/{args.job_id}_{args.prompt}_{args.output_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from time import time

time_counter = 0

counter = 0
with open(f"{args.prompt}.txt", 'r') as file:
    for line_id, line in enumerate(file):
        
        # Each 'line' includes a newline character at the end, you can strip it using .strip()
        prompt = line.strip()
        save_name = '_'.join(prompt.split(' ')).replace('/', '<#>')

        print(prompt)
        # if '/' in prompt:
        #     continue
    
        args.prompt_id = counter
        save_prefix = f"{save_dir}/{args.prompt_id}_{save_name}_common_seed{args.seed}"

        set_seed(args.seed)

        start_time = time()
        # images = pipe(prompt_embeds=auged_prompt_embeds, num_images_per_prompt=4).images
        images = pipe(prompt, num_images_per_prompt=args.batch_size, num_inference_steps=args.num_inference_steps).images
        image = images[0]
        end_time = time()
        print(end_time - start_time)
        try:
            image.save(f"{save_prefix}.png")
            print("image saved at: ", f"{save_prefix}.png")
        except:
            print(f"save at {save_prefix} failed")
            image.save(f"{save_dir}/{args.prompt_id}.png")
            print("image saved at: ", f"{save_dir}/{args.prompt_id}.png")
            continue
        
        if line_id == 0:
            continue
        time_counter += end_time - start_time
        counter += 1
        if counter >= args.counter_exit:
            break

# print(time_counter / counter)
# print(time_counter / counter * (args.counter_exit / 5) / args.batch_size)
