# Import Packages here
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch.nn.functional as F
import torchvision
import json
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import argparse
from torch.autograd import Variable
import additionalFuncs

arg_parse = argparse.ArgumentParser(description='Train.py')
arg_parse.add_argument('--gpu', dest="gpu", action="store", default="gpu")
arg_parse.add_argument('--data_dir', action="store", default="./flowers/")
arg_parse.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
arg_parse.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
arg_parse.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
arg_parse.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.0005)
arg_parse.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
arg_parse.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=510)
    
ap = arg_parse.parse_args()
mode = ap.gpu
data_root = ap.data_dir
checkpoint_path = ap.save_dir
model_arch = ap.arch
epochs = ap.epochs
learn_rate = ap.learning_rate
dropout = ap.dropout
hidden_layers = ap.hidden_units


def main():

    #Step 1: Transform and Load the training, validation and testing data 
    train_loader, valid_loader, test_loader, train_data, valid_data, test_data = additionalFuncs.transform_load_image(data_root)
    
    print("\ndevice: ", mode, "\nData Directory:", data_root, "\nCheckpoint path: ", checkpoint_path, "\nModel            Architecture: ",model_arch, "\nEpochs: ", epochs, "\nLearning Rate: ", learn_rate, "\nDropout: ",dropout,              "\nHidden Layers: ", hidden_layers)
    print("Step 1 complete!")
    
    #Step 2: Build Training Model
    
    #Load pre-trained model
    model = getattr(models,model_arch)(pretrained=True)
    if not mode:
        return torch.device("cpu")
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #Freeze parameters 
    for param in model.parameters():
        param.requires_grad = False
        input_units = model.classifier[0].in_features
    #Create a new untrained feed-forward network as a classifier, using ReLU activations and dropout
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_layers)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layers,102)),
                          ('output', nn.LogSoftmax(dim=1))
                        ]))

        #Replace vgg16 classifier with new classifier
        model.classifier = classifier  
        #initialize criterion 
        criterion = nn.NLLLoss()
        #Optimize the classifier with learning rate
        optimizer = optim.Adam(model.classifier.parameters(), learn_rate)
    # Train model
    model, optimizer = additionalFuncs.training_model(epochs, model, train_loader, valid_loader, criterion, optimizer, device)
    print("Step 2 complete! i.e. build the training model")
    # Test the model
    additionalFuncs.test_trained_model(model, test_loader, device)
    print("Step 3 complete! i.e. tested the trained model")
    # Save model
    additionalFuncs.save_trained_model(model, train_data, optimizer, checkpoint_path, epochs)
    print("Step 4 complete! i.e. trained model is saved")

if __name__== "__main__":
    main()