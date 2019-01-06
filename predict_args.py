#!/usr/bin/env python3
"""predict_args.py
This script parses the arguments for the predict.py module
"""

__author__ = "Drew Massey"
__version__ = "1"
__created__ = "12/31/2018"

import argparse

def get_args():
    """
    grabs the arguments for the predict cli
    """
    
    parser = argparse.ArgumentParser(
        description="Image Prediction",
        usage="python ./predict.py /path/to/image.jpg checkpoint.pth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('path_to_image',
                        action="store",
                        help='path to the image file')
    
    parser.add_argument('checkpoint_file',
                       action="store",
                       help='path to the checkpoint file')
    
    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='directory to saved checkpoint file')
    
    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int,
                        help='return top k most likely classes.')
    
    parser.add_argument('--category_names',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='path to the file with categories.')
    
    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='use gpu')
    
    parser.parse_args()
    return parser

def main():
    print("CLI Utility for predict.py")
    
if __name__ == '__main__':
    main()