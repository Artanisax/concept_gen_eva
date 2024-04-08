import sys
sys.path.append('..')

import os
import subprocess


def generate(name):
    proc = subprocess.Popen(['python', 
                            './scripts/txt2img_common.py',
                            '--prompt', name,
                            '--counter_exit', '2',
                            '--batch_size', '2'])
    proc.communicate()
    print(f'generate {name} finished')


def evaluate(name):
    proc = subprocess.Popen(['python', 
                            './src/evaluation/Q16/main/clip_classifier/classify/inference_images.py',
                            '--input_folder', f'results/{name}',
                            '--output_folder', name])
    proc.communicate()
    print(f'evaluate {name} finished')


def main():
    dataset_root = './datasets/txts'
    file_list = os.listdir(dataset_root)
    for file in file_list:
        name = file[:-4]
        generate(name)
        evaluate(name)



if __name__ == '__main__':
    main()