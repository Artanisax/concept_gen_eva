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
    pass


def main():
    dataset_root = './datasets/txts'
    file_list = os.listdir(dataset_root)
    for file in file_list:
        name = file[:-4]
        generate(name)
        evaluate(name)



if __name__ == '__main__':
    main()