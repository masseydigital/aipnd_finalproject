#!/usr/bin/env python3
"""train_args.py
This script performs training on a specified model.
"""

__author__ = "Drew Massey"
__version__ = "1"

import argparse

"""Supported Architectures"""
arch = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'densenet121',
    'densenet169',
    'densenet161',
    'densenet201'
]

def get_args():
    """
    """
    parser = argparse.ArgumentParser(
    description="Train and save an image classification model.",
    usage="python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units [6272, 1568] --epochs 3",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('data_directory', action="store")
    
    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='Directory to save training checkpoint file')
    
    parser.add_argument('--save_name',
                        action="store",
                        default="checkpoint.pth",
                        dest='save_name',
                        type=str,
                        help="Checkpoint file name")
    
    parser.add_argument('--categories_json',
                        action="store",
                        default="cat_to_name.json",
                        type=str,
                        help="Path to file containing the categories")
    
    parser.add_argument('--arch',
                        action="store",
                        default="vgg16",
                        type=str,
                        help="Supported architectures: {}".format(arch))
    
    parser.add_argument('--gpu',
                        action="store_true",
                        default=False,
                        dest="use_gpu",
                        help="Use GPU")
    
    hp = parser.add_argument_group('hyperparameters')
    
    hp.add_argument('--learning_rate', '-lr',
                    action="store",
                    default=0.001,
                    type=float,
                    help='Learning rate')
    
    hp.add_argument('--hidden_units', '-hu',
                    action="store",
                    dest="hidden_units",
                    default=[6272, 1568],
                    type=int,
                    nargs='+',
                    help='Hidden layer units')
    
    hp.add_argument('--epochs',
                    action="store",
                    dest="epochs",
                    default=3,
                    type=int,
                    help= 'Epochs')
    
    parser.parse_args()
    return parser

def main():
    print("Command line argument parser")
    
if __name__ == '__main__':
    main()