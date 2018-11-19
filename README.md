#AIPND Final Project
PyTorch transfer learning example developed as part of Udacity's AI Programming with Python Nanodegree program.

##Getting Started
Environment
Python 3.6.5:

Numpy
PyTorch
TorchVision

##Usage 

python train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE]
                [--hidden_features HIDDEN_FEATURES] [--gpu]
                data_dir

### positional arguments:
  data_dir              Directory used to locate source images

### optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory used to save checkpoints
  --arch {vgg19, alexnet}
                        Model architecture to use for training
  --learning_rate LEARNING_RATE
                        Learning rate hyperparameter
  --hidden_features HIDDEN_Features
                        Number of hidden units hyperparameter

  --gpu                 Use GPU for training
 

### Examples
The following will train a vgg model on the GPU:
python train.py flowers --arch VGG --gpu 


## Usage
python predict.py  [-h] [--top_k TOP_K] [--gpu] image_path filepath
                 

### positional arguments:
  image_path            Input image
  filepath              Model checkpoint file to use for prediction

### optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Return top k most likely classes
  --gpu                 Use GPU for prediction

### Examples
The following will return the most likely class using a VGG checkpoint executing on the GPU:
python predict.py flowers/test/14/image_0814.jpg VGG_checkpoint.pth --gpu

