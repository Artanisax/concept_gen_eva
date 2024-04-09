import sys
sys.path.append('..')

import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model_name", type=str, default='CompVis/stable-diffusion-v1-4', help="The name of diffusion model.")
parser.add_argument("--local", type=str, default='', help="?")
parser.add_argument("--dataset_root", type=str, default='datasets/txts')
parser.add_argument("--prompt", type=str, default='i2p', help="The txt filename containing prompts.")
parser.add_argument("--job_id", type=str, default='local', help="The id of job.")
parser.add_argument("--output_name", type=str, default='local', help="The name of output.")
parser.add_argument("--counter_exit", default=2, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_inference_steps", default=50, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--device", default='', type=str)

# Parse the arguments
args = parser.parse_args()

# import pdb ; pdb.set_trace()

# python text2img_common.py --prompt mem_living --output_name common_gen_debug
# python text2img_common.py --model_name stabilityai/stable-diffusion-2 --prompt mem_living --output_name common_gen_debug

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

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
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/localscratch/chenkan4/cache", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)

else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/localscratch/chenkan4/cache", safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"./results/{args.prompt}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from time import time

time_counter = 0

with open(os.path.join(args.dataset_root, f"{args.prompt}.txt"), 'r') as file:
    for line_id, line in enumerate(file):
        if line_id >= args.counter_exit:
            break
        
        # Each 'line' includes a newline character at the end, you can strip it using .strip()
        prompt = line.strip()
        # save_name = '_'.join(prompt.split(' ')).replace('/', '<#>')

        print(prompt)
        # if '/' in prompt:
        #     continue
    
        args.prompt_id = line_id
        save_prefix = "{}/{:05d}".format(save_dir, line_id)

        set_seed(args.seed)

        start_time = time()
        images = pipe(prompt, num_images_per_prompt=args.batch_size, num_inference_steps=args.num_inference_steps).images
        end_time = time()
        print(' %.3f s'%(end_time - start_time))
        for img_id, image in enumerate(images):
            image.save(f"{save_prefix}({img_id}).png")

        time_counter += end_time - start_time

print(' %3f'%time_counter)
