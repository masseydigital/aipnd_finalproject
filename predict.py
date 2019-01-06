#!/usr/bin/env python3
"""predict.py
This script performs training on a specified model.
"""

__author__ = "Drew Massey"
__version__ = "1"

import json
import torch
import predict_args
import warnings

from PIL import Image
from torchvision import models
from torchvision import transforms


"""Global Variables"""
parser = predict_args.get_args()
cli_args = parser.parse_args()

def predict(image_path, model, topk=5):
    model.eval()
    
    model.cpu()
    
    image = process_image(image_path)
    
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output=model.forward(image)
        top_prob,top_labels = torch.topk(output, topk)
    
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.numpy()[0], mapped_classes

def load_checkpoint(device, file='checkpoint.pth'):
    model_state = torch.load(file, map_location=lambda storage, loc: storage)
    
    model = models.__dict__[model_state['arch']](pretrained=True)
    model = model.to(device)
    
    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
           
    pil_image = Image.open(image).convert("RGB")
    
    #rescale proportionally
    if pil_image.width > pil_image.height:
        ratio = float(pil_image.width) / float(pil_image.height)
        new_height = ratio * size[0]
        pil_image = pil_image.resize((size[0], int(floor(new_height))), Image.ANTIALIAS)
    else:
        ratio = float(pil_image.height) / float(pil_image.width)
        new_width = ratio * size[0]
        pil_image = pil_image.resize((int(floor(new_width)), size[0]), Image.ANTIALIAS)
    
    in_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    
    pil_image = in_transforms(pil_image)

    return pil_image

def main():
    """Set up device"""
    device = torch.device("cpu")
    if cli_args.use_gpu:
        device=torch.device("cuda:0")
        
    """Load Categories"""
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)
        
    checkpoint_model = load_checkpoint(device, cli_args.checkpoint_file)
    
    top_prob, top_classes = predict(cli_args.path_to_image, checkpoint_model, cli_args.top_k)
    
    label = top_classes[0]
    prob = top_prob[0]
    
    print('Parameters\n-----------------------')
    
    print(f'Image  : {cli_args.path_to_image}')
    print(f'Model  : {cli_args.checkpoint_file}')
    print(f'Device : {device}')
    
    print('\nPrediction\n---------------------')
    
    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')
    
    print('\nTop K\n--------------------------')
    
    for i in range(len(top_prob)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")
              
if __name__ == '__main__':
    main()