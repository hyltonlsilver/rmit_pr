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
import os
from collections import OrderedDict


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
''''>>>>> LOAD VGG16, FIND PARAMETERS <<<<<'''
model1 = models.vgg16(pretrained= True).to(device)
'''print all model detail : disabled once seen'''
# print(model1)

in_features = model1.classifier[0].in_features
print(f'  classifier must have in_features:{model1.classifier[0].in_features}')

''' freeze model '''
for param in model1.features.parameters():
    param.requires_grad = False


    
# print(type(model1))  

'''>>>>> CLASSIFIER  <<<<<'''

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 102)
        
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
      
        return x
print(type(Classifier())) 
print(f' Classifier using static method\n\n{Classifier()}\n\n')   
model1.classifier = Classifier()
print(f' model using classifier using static method\n {model1}\n\n')

class Classifier_2(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            # print(each)
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


clas_2 = Classifier_2(25088,102,[1024,512,256])
print(f'Classifier using dynamic method:\n {clas_2}\n\n')
model1.classifier =clas_2
print(f' model using classifier using dynamic method {model1}\n\n')
forw = clas_2.forward
print(f'Classifier forward  using dynamic method:\n {forw}\n\n')
# model1.classifier = forw
# print(f' model using classifier using dynamic  forward method {model1}\n\n')

