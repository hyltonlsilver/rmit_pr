# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:31:47 2021

@author: hylton
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import optim


import numpy as np
import helper1

import matplotlib.pyplot as plt

import json

import os
import time








'''>>>>>>>  CUDA  <<<<<<<<<<<'''

'''>>>>> SELECT CUDA IF AVAILABLE <<<<<'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(torch.cuda.device_count())
print(f'CUDA is avalable: {torch.cuda.is_available()}')
print(f'The currrent cuda device number is: {torch.cuda.current_device()}')
print(f' The current CUDA device name is: {torch.cuda.get_device_name(torch.cuda.current_device())}')


'''>>>>> Directories<<<<<'''

'''top level directory will be different in their environment'''
data_dir = r'C:\Users\hylto\Documents\Python Scripts\Machine_Learning\main_p\flowers\tryf'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'





'''>>>>> TRANSFORM CODE <<<<<'''

'''image transforms added to training images to reduce overfitting'''
transform_train = transforms.Compose([transforms.ToTensor(),
transforms.Normalize( mean = [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),transforms.Resize(256),transforms.RandomResizedCrop(224),
transforms.RandomRotation(30),transforms.RandomVerticalFlip(),transforms.RandomGrayscale(p=0.1)])

'''image transforms NOT added to validation and testing sets'''
transform_val = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize( mean = [0.485, 0.456, 0.406] , std =[0.229, 0.224, 0.225]),
                              transforms.Resize(256),transforms.CenterCrop(224)])



transform_test = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize( mean = [0.485, 0.456, 0.406] , std =[0.229, 0.224, 0.225]),
                              transforms.Resize(256),transforms.CenterCrop(224)])





'''>>>>> DATASET CODE <<<<<'''

training_data = datasets.ImageFolder(train_dir , transform=transform_train)
validation_data = datasets.ImageFolder(valid_dir , transform=transform_val)
test_data = datasets.ImageFolder(test_dir  , transform=transform_test)



'''>>>>> DATA LOADER CODE <<<<<'''

training_loader = torch.utils.data.DataLoader(training_data, batch_size= 64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


'''>>>>> UNPACK JSON and FIND CAT LEN <<<<<'''

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    class_num = len(cat_to_name)
    
    
'''load model vgg16  and assign to model1'''
model1 = models.vgg16(pretrained= True).to(device)
'''print all model detail : disabled once seen'''
#print(model1)


''' freeze model '''
for param in model1.features.parameters():
    param.requires_grad = False




'''use CUDA. if no CUDA this line will be ignored'''


'''define my classifer and replace the medels classifier with it'''''
classifier =nn.Sequential(nn.Linear(4096,128),
                                  nn.ReLU(),
                                  nn.Linear(128,102),
                                  nn.LogSoftmax(dim=1))
      
                                  
model1.classifier = classifier   
model1.cuda()
print(model1)


'''>>>>> DEFINE LOSS FUNCTION  CODE <<<<<'''

criterion = nn.CrossEntropyLoss()

'''>>>>> DEFINE GRADIENT DESCENT CODE <<<<<'''

optimizer = optim.SGD(model1.classifier.parameters(), lr=0.001)    


epochs = 2

for epoch in range(epochs):
     for my_images, labels in training_loader:
         my_images, labels = my_images.cuda(), labels.cuda()  
          
         logps = model1.forward(my_images)
         loss = criterion(logps, labels)
        
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()


     print(loss.item())
         # running_loss += loss.item()