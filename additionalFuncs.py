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

#-----------------------------------Train Functions--------------------------------------------------------------
#Function to transform and load data
def transform_load_image(root):

    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # transform the training data set by rotation, cropping, flipping and normalizing the means and standard deviations
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # transform the validation data set by resizing, cropping and normalizing the means and standard deviations
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # transform the test data set by resizing, cropping and normalizing the means and standard deviations
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    
    # Pass transforms in the datasets, then run them to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Load the transformed data using dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = True)
    
    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data 

# Function to validate the trained model
def validation(model, valid_loader, criterion, device):
    valid_loss = 0
    accuracy = 0    
    # change model to work with cuda
    model.to(device)
    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(valid_loader):  
        # Change images and labels to work with cuda
        images, labels = images.to(device), labels.to(device)
        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()  
    return valid_loss, accuracy
# Function to build the training model
def training_model(epochs, model, train_loader, valid_loader, criterion, optimizer, device):
    steps = 0
    print_every = 40
    model.to(device)
    for e in range(epochs):
        running_loss = 0    
        # Iterating over data to carry out training step
        for ii, (images, labels) in enumerate(train_loader):
            steps += 1
            images, labels = images.to(device), labels.to(device)     
            # zeroing parameter gradients
            optimizer.zero_grad()       
            # Forward and backward passes
            images=images.float()
            outputs = model.forward(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            #track the training loss
            running_loss += train_loss.item()
        
            # Carrying out validation step
            if steps % print_every == 0:
                # setting model to evaluation mode during validation
                model.eval()
            
                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, device)
                
                    print("Epoch: {}/{} | ".format(e+1, epochs),
                         "Training Loss: {:.4f} | ".format(running_loss/print_every),
                         "Validation Loss: {:.4f} | ".format(valid_loss/len(valid_loader)),
                         "Validation Accuracy: {:.4f}".format(accuracy/len(valid_loader)))
            
                running_loss = 0          
            # Turning training back on
                model.train()
    print("\nTraining model is built!!")
    return model, optimizer
#Function to test the trained model with test data
def test_trained_model(model, test_loader, device):
# validation on the test set
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # Get probabilities
            outputs = model(images)
            # Turn probabilities into predictions
            _, predicted = torch.max(outputs.data, 1)
            # Total number of images
            total += labels.size(0)
            # Count number of cases in which predictions are correct
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return
#Function to save the model into checkpoint file used for rebuilding model
def save_trained_model(model, train_data, optimizer, save_dir, epochs):
    # TODO: Save the checkpoint 
    # Dictionary checkpoint contains architecture parameters 
    checkpoint = {'classifier': model.classifier,
                'epoch': epochs,  
                'optimizer': optimizer.state_dict(), 
                'state_dict': model.state_dict(), 
                'class_to_idx': train_data.class_to_idx 
                }
    return torch.save(checkpoint, save_dir)

#---------------------------------------Predict Functions-------------------------------------------------------------
# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_path, model_arch):
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models,model_arch)(pretrained=True)
    model.eval()
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict']) 
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    return model
# Function to process image
def process_image(image):
    #Perform transformations, convert to tensor and normalize (easier)
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    #Open image and apply transformation
    pil_image = Image.open(image)
    pil_image = transform(pil_image)
    #Convert to numpy array for return
    np_image = np.array(pil_image)          
    return np_image
#Function to predict classes for loaded images
def predict(image_path, model, top_k):
        #model = load_checkpoint(model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        np_image = process_image(image_path) #numpy array returned
        torch_image = torch.from_numpy(np_image).to(device) #convert to cuda tensor
        torch_image = torch_image.unsqueeze_(0).float() #returns float 'cuda' tensor of single dimension (1 column)  
    
        with torch.no_grad(): 
            output = model.forward(torch_image) 
            ps = torch.exp(output) 
    
        #taking top 5 probabilities and their indices 
        probs, indices = torch.topk(ps, top_k)
    
        #invert class_to_idx
        inv_class_to_idx = {index: cls for cls, index in model.class_to_idx.items()}
    
        classes = []
        for index in indices.cpu().numpy()[0]: #iterating through indices
            classes.append(inv_class_to_idx[index])
    
        return probs.cpu().numpy()[0], classes