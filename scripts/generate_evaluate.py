import sys
sys.path.append('..')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import subprocess


def generate(name):
    proc = subprocess.Popen(['python', 
                            './scripts/txt2img_common.py',
                            '--prompt', name,
                            '--counter_exit', '5',
                            '--batch_size', '2',
                            '--device', os.environ['CUDA_VISIBLE_DEVICES']])
    proc.communicate()
    print(f'generate {name} finished')


def evaluate(name):
    proc = subprocess.Popen(['python', 
                            './src/evaluation/Q16/main/clip_classifier/classify/inference_images.py',
                            '--input_folder', f'results/{name}',
                            '--output_folder', name,
                            '--device', os.environ['CUDA_VISIBLE_DEVICES']])
    proc.communicate()
    print(f'evaluate {name} finished')


def main():
    dataset_root = './datasets/txts'
    file_list = sorted(os.listdir(dataset_root))
    print(file_list)
    # for file in file_list:
    #     name = file[:-4]
    #     generate(name)
    #     evaluate(name)


if __name__ == '__main__':
    main()