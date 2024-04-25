import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str)
parser.add_argument("--input_folder", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()

import os
import shutil
from tqdm import tqdm
import pandas

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

df = pandas.read_csv(args.input_csv, names=['class', 'hard', 'score', 'filename'])
for filename in tqdm(df['filename']):
    shutil.copy(os.path.join(args.input_folder, filename), os.path.join(args.output_folder, filename))
