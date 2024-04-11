import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--input_folder", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()

import os
from tqdm import tqdm
import pandas as pd
from nudenet import NudeDetector

all_labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
NSFW_labels = [
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
]

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)


nude_detector = NudeDetector()
data = pd.DataFrame(columns=['class'] + all_labels, dtype=float)
for filename in tqdm(sorted(os.listdir(args.input_folder))):
    df = pd.DataFrame(nude_detector.detect(os.path.join(args.input_folder, filename)))
    if not df.empty:
        df = pd.DataFrame([list(df['score'])], columns=df['class'], index=[filename]).T.groupby(level=0).max().T
        # print(data)
        # print(df)
        data = pd.concat([data, df])

# data[NSFW_labels].dropna(how='all').round(4) \
#     .to_csv(os.path.join(args.output_folder, f"{args.name}.csv"), index_label='filename')
    
data.dropna(how='all').round(4) \
    .to_csv(os.path.join(args.output_folder, f"{args.name}_all.csv"), index_label='filename')
