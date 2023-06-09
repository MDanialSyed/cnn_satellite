"""Functions for training the CNN model."""

import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import VGG11_BN_Weights
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import time
import copy

import os


torch.manual_seed(27)
np.random.seed(27)

#### Disclaimer: CNN code is adapted from Pytorch tutorial by Nathan Inkawhich ####
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

# Top level data directory.
data_dir = './bld/python/data'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception
model_name = "vgg"

# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 8


# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def initialize_model(model_name, num_classes, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific. Removed other choices for simplicity. 
    model_ft = None
    input_size = 0
    model_name == "vgg"
    model_ft = models.vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224
    
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        if epoch > 2:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def identify_data(data_transforms):

    # Create training and validation datasets and dataloaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                           batch_size=batch_size, shuffle=True, 
                                                           num_workers=4) for x in ['train', 'val']}
    return dataloaders_dict


def model_execute(epochs):
    
    # Specify number of epochs
    num_epochs = epochs
    
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract)

    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(input_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]),
        'val': transforms.Compose([transforms.Resize(input_size),
                                   transforms.CenterCrop(input_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ]),
    }

    # Create training and validation datasets and dataloaders
    dataloaders_dict = identify_data(data_transforms)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated depending on feature_extract
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate vgg
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs)
    
    return model_ft, hist
    
