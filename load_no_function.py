# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 18:12:48 2021

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
import seaborn as sns
import json
import os
import time
import PIL
from PIL import Image
import glob
from collections import OrderedDict

'''specific code for  my machine as it crashes otherwise - so not transfer to RMIT '''

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_dir = r'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

'''>>>>> TRANSFORM <<<<<'''
'''image transforms added to training images to reduce overfitting'''
transform_train = transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),
transforms.RandomRotation(30),transforms.RandomVerticalFlip(),
transforms.ToTensor(),
transforms.Normalize( mean = [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])])


'''image transforms NOT added to validation and testing sets'''
transform_vt = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                transforms.Normalize( mean = [0.485, 0.456, 0.406] , std =[0.229, 0.224, 0.225]),
                              ])

'''>>>>> DATASETS <<<<<'''
training_data = datasets.ImageFolder(train_dir , transform=transform_train)
validation_data = datasets.ImageFolder(valid_dir , transform = transform_vt)
test_data = datasets.ImageFolder(test_dir  , transform = transform_vt)


'''>>>>> DATA LOADERS <<<<<'''
training_loader = torch.utils.data.DataLoader(training_data, batch_size= 64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

       
  









model_select = 'alexnet'
output_size = 102
hidden_layers = [1024,512,256,128,120]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mod_import(t_mod):
   
   
    if t_mod == 'vgg16':
     model = models.vgg16(pretrained=True)
     input_number = 25088
     print(f'Model {t_mod} selected')
     return model, input_number
 
    elif t_mod == 'densenet121':
     model = models.densenet121(pretrained=True)
     input_number = 1024
     print(f'Model{t_mod} selected')
     return model, input_number
 
    elif t_mod == 'alexnet':
     model = models.alexnet(pretrained = True) 
     input_number = 9216
     print(f'Model{t_mod} selected')
     
    else: print('Invalid model selected')
    return model, input_number

    for param in model.features.parameters():
     param.requires_grad = False
     
    print(f' model selected {model} with in parameters of {input_number}')
    return model.to(device), input_number



model, input_size =mod_import(model_select)
'''using fc model  from lessons'''  


    


loaded_cp = torch.load(r'checkpoint_1.pth')



# def load_checkpoint(device, filepath = r'checkpoint_1.pth'):
      
#      loaded_cp = torch.load(filepath)
    
     
        
#      for param in model.parameters():
#         param.requires_grad = False
    
#         model.class_to_idx = loaded_cp['class_to_idx']
    
    
#      model.classifier = loaded_cp['classifier']
#      model.load_state_dict(loaded_cp['state_dict'])
    
#      model.class_to_idx = loaded_cp['class_to_idx']
    
    
#      return model.to(device)

# model2= load_checkpoint(device, 'checkpoint_1.pth')
# model2.to(device)
# print(model2.classifier)
# print(f'\n\n\n {model2}')



