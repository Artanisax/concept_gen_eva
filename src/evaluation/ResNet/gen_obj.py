import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--object', type=str, required=True)
parser.add_argument('--prompt_id', type=int, required=True)
args = parser.parse_args()
obj_ = args.object
obj = obj_.replace('_', ' ')
prompt_id = args.prompt_id

import os
import torch
from diffusers import StableDiffusionXLPipeline

device = 'cuda'


specific = {
    "tench": "tench in the tank meet tench in the ocean",
    "English springer": "an agile English springer with a stick in its mouth",
    "cassette player": "cassette player in an old-school studio",
    "chain saw": "chain saw holding by a logger",
    "church": "a model church on a table where children playing around",
    "French horn": "a French horn been played in an orchestra",
    "garbage truck": "a garbage truck in heavy traffic",
    "gas pump": "a gas pump being used",
    "golf ball": "a golf ball on the grass, about to be hit",
    "parachute": "an opened parachute in high sky"
}
assert obj in specific.keys()


def main():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    gen = torch.Generator(device)

    save_dir = f'datasets/{obj_}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    prompts = [
        f"a picture of {obj}",
        f"a photo of {obj}",
        f"an artwork of {obj}"
        f"imagination containing {obj}",
        specific[obj],
    ]

    prompt = prompts[prompt_id]
    for id in range(2):
        img_id = prompt_id * 2 + id
        gen.manual_seed(img_id)
        save_path = os.path.join(save_dir, f'{img_id}.png')
        pipe(prompt, gen=gen).images[0].save(save_path)
        print(f"Saved to {save_path}.")

main()