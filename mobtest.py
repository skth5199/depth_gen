import os
import glob
import time
from PIL import Image
import numpy as np
import PIL
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
import torchvision.models as models
import cv2
from Mobile_model import Model

loc_img="test"

depth_dataset = DepthDataset(root_dir=loc_img)
fig = plt.figure()
len(depth_dataset)
for i in range(len(depth_dataset)):
    sample = depth_dataset[i]
    print(i, sample['image'].size)
    plt.imshow(sample['image'])
    plt.figure()

    if i == 6:
        plt.show()
        break

depth_dataset = DepthDataset(root_dir=loc_img,transform=transforms.Compose([ToTensor()]))
batch_size=1
train_loader=torch.utils.data.DataLoader(depth_dataset, batch_size)
dataiter = iter(train_loader)
images = dataiter.next()

model = Model().cuda()
#model = nn.DataParallel(model)
#load the trained model
model.load_state_dict(torch.load('49.pth'))
model.eval()

for i,sample_batched1  in enumerate (train_loader):
    image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
    
    outtt=model(image1 )
    x=outtt.detach().cpu().numpy()
    img=x.reshape(240,320)
    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    plt.imsave('genimg/1%d_depth.jpg' %i, resized, cmap='inferno') 
    
    s_img=sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0)
    plt.imsave('genimg/1%d_image.jpg' %i, s_img) 