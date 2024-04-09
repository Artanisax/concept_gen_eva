import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()

import os
from tqdm import tqdm
import pandas
from nudenet import NudeDetector

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

nude_detector = NudeDetector()
for filename in tqdm(os.listdir(args.input_folder)):
    pandas.DataFrame(nude_detector.detect(os.path.join(args.input_folder, filename))).to_csv(
        os.path.join(args.output_folder, filename[:-3]+'csv'), index=False)
    