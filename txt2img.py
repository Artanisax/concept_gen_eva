import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="an integer to be processed")
parser.add_argument("--local", type=str, default='', help="The scale of noise offset.")
parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="The scale of noise offset.")
parser.add_argument("--job_id", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--output_name", type=str, default='local', help="The scale of noise offset.")
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
import safetensors

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

model_id = args.model_name
device = "cuda"

if args.model_name == "stabilityai/stable-diffusion-2":
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/localscratch/renjie/cache/", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float32)

elif args.model_name == "CompVis/stable-diffusion-v1-4":

    data = safetensors.torch.load_file('/egr/research-dselab/renjie3/renjie/NeurIPS24_concept_removal/diffusers/examples/textual_inversion/textual_inversion_teddy_10step_lr5e-3/learned_embeds-steps-10.safetensors')

    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer", cache_dir="/localscratch/renjie/cache2/")
    text_encoder = CLIPTextModel.from_pretrained(
            args.model_name, subfolder="text_encoder", cache_dir="/localscratch/renjie/cache2/"
        )

    placeholder_tokens = ["<rj-begin>", "<rj-end>"]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for i, token_id in enumerate(placeholder_token_ids):
            # import pdb ; pdb.set_trace()
            token_embeds[token_id] = data['<cat-toy>'][i].clone().to(torch.float32)

    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/localscratch/renjie/cache2/", safety_checker=None, tokenizer=tokenizer, text_encoder=text_encoder, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"./results/{args.job_id}_{args.prompt}_{args.output_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from time import time

import numpy as np
import random

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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

        set_seed(args.seed)
    
        args.prompt_id = counter
        save_prefix = f"{save_dir}/{args.prompt_id}_{save_name}_seed{args.seed}"

        # auged_prompt_embeds = torch.load("/egr/research-dselab/renjie3/renjie/diffusion/ECCV24_diffusers_memorization/diffusers/examples/text_to_image/dm-memorization-public/tensor.pt")

        # import pdb ; pdb.set_trace()

        start_time = time()
        # images = pipe(prompt_embeds=hidden_states_org, num_images_per_prompt=args.batch_size, num_inference_steps=args.num_inference_steps).images
        images = pipe(prompt, num_images_per_prompt=args.batch_size, num_inference_steps=args.num_inference_steps).images
        image = images[0]
        end_time = time()
        print(end_time - start_time)
        try:
            image.save(f"{save_prefix}.png")
            print("image saved at: ", f"{save_prefix}.png")
        except:
            print(f"save at {save_prefix} failed")
            continue
        
        if line_id == 0:
            continue
        time_counter += end_time - start_time
        counter += 1
        if counter >= args.counter_exit:
            break

# print(time_counter / counter)
# print(time_counter / counter * (args.counter_exit / 5) / args.batch_size)
