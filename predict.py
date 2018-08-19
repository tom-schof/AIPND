#I modified my jupyter notebook from part one to create the part 2 project as well as the image classifier lab and argparse
# tutorial found at https://docs.python.org/2/howto/argparse.html. 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.optim import lr_scheduler
from PIL import Image
import json
import time
import os
import copy

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def get_input_args():
   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type=str, help='Enter the image path: ')
    parser.add_argument('filepath', type=str, help='Load checkpoint to rebuild model:')
    parser.add_argument('--top_k', default=3, type=int, help=' top_k results')
    parser.add_argument('--gpu', default=True, action='store_true', help='use GPU? (True/False')
    
    return parser.parse_args()



# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['arch']
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_features'])),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(checkpoint['hidden_features'], (checkpoint['hidden_features'] // 8))),
                              ('fc3', nn.Linear((checkpoint['hidden_features'] // 8), 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))

    
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    #model.class_to_idx(checkpoint['class_to_idx'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_transforms = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = image_transforms(pil_image)
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image
    
    

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image_tensor = torch.FloatTensor([process_image(image_path)])
    
    if gpu and torch.cuda.is_available():
        model.cuda()
        image_tensor = image_tensor.cuda()
        print('using gpu')
    else:
        print('using cpu')
 
    model.eval()
    logits = model.forward(image_tensor)
    probs= F.softmax(logits.data, dim=1)
    

    data = torch.topk(probs, topk, sorted=True)
    probs = np.array(data[0])
    probs = probs[0]
    classes = np.array(data[1])
    classes = classes[0]
    
    names = []
    for x in np.nditer(classes):
        names.append(cat_to_name[str(x)])
        
    return probs, names

def main():
    in_args = get_input_args()
    loaded_model = load_checkpoint(in_args.filepath)
    probs, classes = predict(in_args.image_path, loaded_model, in_args.top_k, in_args.gpu)
    print('most likely flowers: ', classes)
    print('probabilities: ', probs)

if __name__ == '__main__':
    main()
    print('predict.py main method')

  