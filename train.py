#I modified my jupyter notebook from part one to create the part 2 project as well as the image classifier lab and argparse tutorial found at https://docs.python.org/2/howto/argparse.html
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
import os
import json
import time
import copy

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def get_input_args():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('data_dir', type=str, help='Enter the Data Directory:')
    parser.add_argument('--save_dir', default='checkpoint.pth', help='Where to save the checkpoint:')
    parser.add_argument('--arch', default='VGG', help='Network Architecture:(VGG or alexnet)')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Enter the Learning Rate:' )
    parser.add_argument('--hidden_features', default='4096', type=int, help='Enter the number of hidden layers:')
    parser.add_argument('--gpu', default=True, action='store_true', help='use GPU? (True/False')
    
    return parser.parse_args()

def prepare_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

#Defines transforms for the training, validation, and testing sets

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    #Load the datasets with ImageFolder
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    #Using the image datasets and the trainforms, defines the dataloaders
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=32, shuffle=True)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

   

    return dataloaders, dataset_sizes, class_names


    
def model_select(arch):    
# Select Model Architecture and Build and train your network
    model = ''
    input_features = 0
    if arch == 'VGG':
        model = models.vgg19(pretrained = True)
        input_features = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_features = 9216
    else:
        print('Invalid model architecture. Try VGG or alexnet')


    for param in model.parameters():
        param.requires_grad = False
    


    return model, input_features

def create_classifier(model, input_features, hidden_features, learning_rate):
    #print('input:' , input_features.type())
    #print('hidden: ', hidden_features.type())
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_features, hidden_features)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_features, (hidden_features // 8))),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear((hidden_features // 8), 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    model.classifier = classifier
    # Create the network, define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    num_epochs = 11
    
    return model, criterion, optimizer, exp_lr_scheduler, num_epochs

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, gpu):
    since = time.time()
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('using gpu')
    else:
        device = torch.device("cpu")
        print('using cpu')
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if gpu and torch.cuda.is_available():
                        device = torch.device("cuda:0")    
                    else:
                        device = torch.device("cpu")
                    model.to(device)    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

 #model = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=11)

# TODO: Do validation on the test set
def test_validation(model, dataloaders, gpu):  
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in dataloaders['test']:
            if gpu and torch.cuda.is_available():
                device = torch.device("cuda:0")    
            else:
                device = torch.device("cpu")
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.to(device)        
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
    # TODO: Save the checkpoint 
def save_model(save_dir, model, dataloaders, input_features, hidden_features, arch, learning_rate, optimizer):
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    checkpoint = {'input_size': input_features,
                  'hidden_features' : hidden_features,
                  'learning_rate' : learning_rate,
                  'output_size' : 102,
                  'epoch' : 11, 
                  'arch': arch,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_dict':optimizer.state_dict(),
                  'state_dict': model.state_dict()
                 }

    torch.save(checkpoint, save_dir)
  
    return 'model saved'

def main():
    in_args = get_input_args()
    print("ran main")
    dataloaders, dataset_sizes, class_names = prepare_data(in_args.data_dir)
    model_arch, inputs = model_select(in_args.arch)
    model, criterion, optimizer, exp_lr_scheduler, num_epochs = create_classifier(model_arch, inputs,\
                                                                in_args.hidden_features, in_args.learning_rate)
    trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs, in_args.gpu)
    test_validation(trained_model, dataloaders, in_args.gpu)
    save_model(in_args.save_dir, trained_model, dataloaders, inputs, in_args.hidden_features, model_arch, in_args.learning_rate, optimizer)
    
if __name__ == '__main__':
    main()
    print('train.py main method')