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

arg_parse = argparse.ArgumentParser(description='Predict.py')
arg_parse.add_argument('--image_path', action="store", default="../ImageClassifier/flowers/test/28/image_05230.jpg")
arg_parse.add_argument('--cat_name_dir', action="store", dest="cat_name_dir", default = "cat_to_name.json")
arg_parse.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
arg_parse.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
arg_parse.add_argument('--top_k', dest="topk", action="store", default=3, type = int)
arg_parse.add_argument('--gpu', dest="gpu", action="store", default="gpu")

ap = arg_parse.parse_args()
mode = ap.gpu
image_path = ap.image_path
checkpoint_path = ap.save_dir
cat_dir_name = ap.cat_name_dir
model_arch = ap.arch
top_k = ap.topk
with open(cat_dir_name, 'r') as f:
    cat_to_name = json.load(f)
    
def main():                      
    # Function that loads a checkpoint and rebuilds the model
    load_model = additionalFuncs.load_checkpoint(checkpoint_path, model_arch)
    print("Print the model data", load_model)

    #Predict the probabilities and classes                   
    probs, classes = additionalFuncs.predict(image_path, load_model, top_k)
    print(probs)
    print(classes)
    names = []
    for i in classes:
        names += [cat_to_name[i]]
    print("flowers ", names)
    # Print name of predicted flower with highest probability
    print(f"This flower is most likely to be a: '{names[0]}' with a probability of {round(probs[0]*100,4)}% ")

if __name__== "__main__":
    main()