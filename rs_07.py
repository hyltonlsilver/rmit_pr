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

''' Operating Parameters '''
num_e =int(input("Epochs: "))
learning_rate = 0.001

'''>>>>> SELECT CUDA IF AVAILABLE <<<<<'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(torch.cuda.device_count())
print(f'CUDA is avalable: {torch.cuda.is_available()}')
print(f'The currrent CUDA device number is: {torch.cuda.current_device()}')
print(f'The current CUDA device name is: {torch.cuda.get_device_name(torch.cuda.current_device())}')


'''>>>>> Directories<<<<<'''

'''top level directory will be different in their environment'''
data_dir = r'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
test_image  = train_dir + r'\1\image_06734.jpg'



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


'''>>>>> UNPACK JSON and FIND CAT LEN <<<<<'''
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    class_num = len(cat_to_name)
        
    
''''>>>>> LOAD VGG16, FIND PARAMETERS <<<<<'''
model1 = models.vgg16(pretrained= True).to(device)
'''print all model detail : disabled once seen'''
# print(model1)

in_features = model1.classifier[0].in_features
print(f'  classifier must have in_features:{model1.classifier[0].in_features}')

''' freeze model '''
for param in model1.features.parameters():
    param.requires_grad = False
    
   

'''>>>>> CLASSIFIER  <<<<<'''

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 102)
        
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
    
 
model1.classifier = Classifier()

model1.to(device)


'''>>>>> DEFINE LOSS FUNCTION <<<<<'''
criterion = nn.NLLLoss()

'''>>>>> DEFINE OPTIMISER <<<<<'''
optimizer = optim.Adam(model1.classifier.parameters(), lr= learning_rate)    


'''||||||||||||||||| >>>>>>>>>>TRAIN THE MODEL <<<<<<<<<< |||||||||||||||||||||||||||||||||'''

epochs = num_e
train_losses, test_losses, accuracy_hist = [], [], []
start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    model1.train()
    start_epoch = time.time()
    accum_t_loss = 0 
    it_num =0
    
    model1.train()
    
    for my_images, labels in training_loader:
        it_num +=1
        start_image = time.time()
        my_images, labels = my_images.to(device), labels.to(device) 
        
        '''forward pass''' 
        logps = model1(my_images)
        '''calculate loss'''
        loss = criterion(logps, labels)
        # print(f'type  training loss {type(loss)}') 
        optimizer.zero_grad()
        '''back and optimiser'''
        loss.backward()
        optimizer.step()
        # print(f'type training loss item {type(loss.item())}')
        accum_t_loss += loss.item()
                
        '''VALIDATE'''
    else:
         training_loss = accum_t_loss/len(training_loader)
        
         
         accum_test_loss = 0
         test_correct = 0
         
         with torch.no_grad():
             model1.eval()
             accuracy = 0
             for my_images,labels in validation_loader:
                my_images, labels = my_images.to(device), labels.to(device)  
                
                '''forward pass'''
                logps = model1(my_images)
                '''calculate loss'''
                test_loss = criterion(logps, labels)
                accum_test_loss += test_loss.item()
                ps = torch.exp(logps)
                
                '''test accuracy'''
                topP, top_class = ps.topk(1,dim=1)
                correct = top_class == labels.view(*top_class.shape)
                # print(f'correct 1st: {correct}\n')
                test_correct += correct.sum().item()
                correct = correct.type(torch.FloatTensor)
                accuracy += torch.mean(correct).item()
                # print(f'correct: {correct}\n')
                # print(f'correct item: {torch.mean(correct).item()}\n')
                # print(f'accracy:  {accuracy}')
                # print(accuracy)
         model1.train()
         
         end_time = time.time() - epoch_start_time
         running_time = time.time() - start_time
         print(f'Epoch duraion: {end_time:.2f}. Running  Time: {running_time:.2f}' )      
                #
                
         # print(f'bottom accuracy:{accuracy}, Len: {len(validation_loader)} e accuracy: {e_accuracy}')      
         accuracy_hist.append(accuracy/len(validation_loader)) 
         train_losses.append(accum_t_loss/len(training_loader))
         test_losses.append(accum_test_loss/len(validation_loader))
      
         
            
         print(f'Epoch:{ epoch+1}\nTraining Loss: {train_losses[-1]:.3f}')
         print(f'Test Loss: {test_losses[-1]:.3f}')
         print(f'Accuracy:{accuracy_hist[-1]:.2%}')
            # print("Epoch: {}/{}.. ".format(e+1, epochs),
            #       "Training Loss: {:.3f}.. ".format(train_losses[-1]),
            #       "Test Loss: {:.3f}.. ".format(test_losses[-1]),
            #       "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
plt.plot(train_losses, label='Training loss')
# print(f)
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)    



'''>>>> CHECK ACCURACY ON TEST SET <<<<<<'''
def check_accuracy():
    accum_check_loss = 0
    check_test_correct = 0
    for e in range (5):
        print("checking accuracy")
        with torch.no_grad():
                     model1.eval()
                     accuracy = 0
                     for my_images,labels in test_loader:
                        my_images, labels = my_images.to(device), labels.to(device)  
                        
                        '''forward pass'''
                        logps = model1(my_images)
                        '''calculate loss'''
                        test_loss = criterion(logps, labels)
                        accum_check_loss += test_loss.item()
                        ps = torch.exp(logps)
                        
                        '''test accuracy'''
                        topP, top_class = ps.topk(1,dim=1)
                        correct = top_class == labels.view(*top_class.shape)
                        # print(f'correct 1st: {correct}\n')
                        check_test_correct += correct.sum().item()
                        correct = correct.type(torch.FloatTensor)
                        accuracy += torch.mean(correct).item()
                        # print(f'correct: {correct}\n')
                        # print(f'correct item: {torch.mean(correct).item()}\n')
                        # print(f'accracy:  {accuracy}')
                        # print(accuracy)
            
             
            
                
        accuracy_hist.append(accuracy/len(validation_loader)) 
        train_losses.append(accum_t_loss/len(training_loader))
        test_losses.append(accum_test_loss/len(validation_loader))
          
             
           
        print(f' TEST Epoch:{ epoch+1}\nTraining Loss: {train_losses[-1]:.3f}')
        print(f'TEST Test Loss: {test_losses[-1]:.3f}')
        print(f'TEST Accuracy:{accuracy_hist[-1]:.2%}')
                # print("Epoch: {}/{}.. ".format(e+1, epochs),
                #       "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                #       "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                #       "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
        plt.plot(train_losses, label='Training loss')
        # print(f)
        
        plt.legend(frameon=False)    

check_accuracy()



''' >>>>>>>>>>>>>>>   SAVE MODEL and CHECKPOINT   <<<<<<<<<<<'''

def model_save():
    
    model1.class_to_idx = training_data.class_to_idx
    
    # print(f'model class to index\n\n{model1.class_to_idx}\n\n\n')
    model1.cpu()
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 1024)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(1024, 512)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(512, 256)),
                              ('relu', nn.ReLU()),
                              ('fc4', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return torch.save({
                'state_dict': model1.state_dict(), 
                'optimiser':optimizer.state_dict(),
                'classifier':classifier,
                'class_to_idx': model1.class_to_idx}, 
                'checkpoint.pth')
    
model_save()   







''' >>>>>>>>>>>>>>>  LOAD MODEL and CHECKPOINT   <<<<<<<<<<<'''

def load_model(checkpoint_path):
    loaded_cp = torch.load(checkpoint_path)
    
    model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
        model.class_to_idx = loaded_cp['class_to_idx']
    
    #    
    # model.classifier = classifier
    model.classifier = loaded_cp['classifier']
    model.load_state_dict(loaded_cp['state_dict'])
    
    print('saved model1 uploaded')
    return model1
    
model1 =load_model('checkpoint.pth')



'''' >>>>>>>> PROCESS IMAGE <<<<<<<<<<''' 

def process_image(image):
    data= Image.open(image)
   
    image = data.resize((256,256)).crop((16,16,240,240))
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    # print(torch.FloatTensor(image))
   
    return torch.FloatTensor(image)

# test_process = process_image(test_image)

'''' >>>>>>>> TEST PROCESS IMAGE <<<<<<<<<<''' 

def imshow(image, ax=None, title=None):

    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()
        image = image.numpy().transpose((1, 2, 0))
# Undo preprocessing
# 
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

# Image needs to be clipped between 0 and 1 or it looks like noise when displayed

        image = np.clip(image, 0, 1)

        ax.imshow(image)

    return ax

# imshow(test_process)
  
''' >>>>>>> CLASS PREDICTIONS   >>>>>>'''

def predict(image_path, model = model1, topk=5):
    
    '''send the image to the process_image function'''
    processed_image = process_image(image_path)
    print(f'\n\nprocessed image type in predict function:{type(processed_image)}\n\n')
    # print(processed_image)
    
    ''' transform image for this fuction'''
    processed_image.unsqueeze_(0)
    
    '''calculate probabilities'''
    probability = torch.exp(model.forward(processed_image))
    
         
    print(f'\n\nprobaility type {type(probability)}\n\n')
    top_probs, top_labs = probability.topk(topk)
    print(f'\n\ntype top probs{type(top_probs)}\n\n')
    print(f'\n\nthis is type top labs {type(top_labs)}\n\n')
    top_probs = top_probs[0].detach().numpy()
    
    
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


prob, classes, flowers = predict(test_image)
print(f'Pobabilty:{prob}')
print(f'Classes: {classes}')
print(f'Flowers: {flowers}')


''' >>>>>> SANITY CHECK <<<<<<<<'''

def plot_solution(image_path, model):
    # Sets up our plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('\\')[1]
    header= cat_to_name[flower_num] # Calls dictionary for name
    # Plot flower
    img = process_image(image_path)
    print(f'image type in plot solution {type(img)}')
    # print(img)
    plt.title(header)
    # ax.imshow(img)
    imshow(img, ax)
    # Make prediction
    top_probs, top_labels, top_flowers = predict(image_path, model) 
    # top_probs = top_probs[0].detach().numpy() #converts from tensor to nparray
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=top_probs, y=top_flowers, color=sns.color_palette()[0]);
    plt.show()

    print(top_probs, top_labels, top_flowers)

plot_solution(test_image, model1)