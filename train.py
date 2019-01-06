#!/usr/bin/env python3
"""train.py
This script performs training on a specified model.
"""

__author__ = "Drew Massey"
__version__ = "1"

import sys
import os
import json
import train_args
import torch

from collections import OrderedDict
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""Global Variables"""
parser = train_args.get_args()
cli_args = parser.parse_args()

# set up our device
device = torch.device("cpu")
if cli_args.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")

# Implement a function for the validation pass
def validation(model, valid_loader, idx):
    valid_correct = 0
    valid_total = 0
    
    # change to cuda
    model.to('cuda')
    
    for ii, (images, labels) in enumerate(valid_loader):
        # Move images and labeles perferred device
        # if they are not already there
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        valid_total += labels.size(0)
        valid_correct += (predicted == labels).sum().item()
    
    print(f"\n\tValidating for Epoch {idx+1}...")
    correct_percent = 0
    
    if valid_correct > 0:
        correct_percent = (100 * valid_correct // valid_total)
    
    print(f'\tAccurately classified {correct_percent:d}% of {valid_total} images./n')

# A function that loads a checkpoint and rebuilds the model
def load_checkpoint(file=cli_args.save_name):
    # Loading weights for CPU model while trained on GP
    model_state = torch.load(file, map_location=lambda storage, loc: storage)
    
    model = models.vgg16(pretrained=True)
    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']
    model.arch = cli_args.arch

    return model

def main():
    #File directories for our data
    data_dir = cli_args.data_directory
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # make sure data directory exists
    if not os.path.isdir(data_dir):
        print('Data directory {} not found'.format(data_dir))
        exit()
        
    # make sure the save directory exists
    if not os.path.isdir(cli_args.save_dir):
        print('Save directory {} does not exists.  Creating...'.format(cli_args.save_dir))
        os.makedirs(cli_args.save_dir)
    
    # load the categories
    with open(cli_args.categories_json, 'r') as file:
        cat_to_name = json.load(file)
    
    output_size = len(cat_to_name)
    print("Images are labels with {} categories.".format(output_size))
    
    # data loader information
    expected_means = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    max_image_size = 224
    batch_size = 64
    
    # define training transform, dataset, and loader
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(max_image_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(expected_means, expected_std)])
    
    valid_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(max_image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(expected_means, expected_std)])
    
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_set = datasets.ImageFolder(valid_dir, transform = valid_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=10, shuffle = True)
    
    model=models.__dict__[cli_args.arch](pretrained=True)
    
    densenet_input = {
        'densenet121' : 1024,
        'densenet169' : 1664,
        'densenet161' : 2208,
        'densenet201' : 1920
    }
    
    # if our architecture is vgg, we need to grab the input_sizes
    if cli_args.arch.startswith("vgg"):
        input_size = model.classifier[0].in_features
    
    # if our architecture is densenet, we need to grab the input_sizes
    if cli_args.arch.startswith("densenet"):
        input_size = densenet_input[cli_args.arch]
        
    # grab our hidden units
    hidden_sizes = cli_args.hidden_units
    
    # grab our output size
    output_size = len(cat_to_name)

    # Prevent back propagation on parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # build our classifier
    #Set our classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu', nn.ReLU()),
        ('logits', nn.Linear(hidden_sizes[1], output_size)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.classifier = classifier

    model = model.to(device)
    
    # training criteria and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=cli_args.learning_rate) 
    
    #train our model
    print_every = 40
    steps = 0
    epochs = cli_args.epochs
    test_total=0
    test_correct=0
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
        
            if steps % print_every == 0:
                avg_loss = f'{running_loss/(e+1):.4f}'
                acc = f'{(test_correct/test_total) * 100:.2f}%'
                print("Epoch: {}/{}.. Test Loss: {}.. Test Accuracy {}.".format(e+1, epochs, avg_loss, acc))

                running_loss = 0
            
                model.train()

        # perform validation on the epoch
        with torch.no_grad():  
            validation(model, valid_loader, e)
                
    # Save the checkpoint 
    checkpoint_file_name = cli_args.save_name

    model.class_to_idx = train_set.class_to_idx

    checkpoint = {
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
        'arch': cli_args.arch
    }

    torch.save(checkpoint, checkpoint_file_name)               

    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)