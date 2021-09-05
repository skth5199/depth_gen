import cv2
import kornia 
import matplotlib
import time
import datetime
import pandas as pd
import numpy as np
import torch
import os
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as Functional  
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils import shuffle
from itertools import permutations

class upsampleBlock(nn.Sequential):
    def __init__(self, skin, opFeature):
        super(upsampleBlock, self).__init__()        
        self.convA = nn.Conv2d(skin, opFeature, kernel_size=3, stride=1, padding=1)
        self.lrA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(opFeature, opFeature, kernel_size=3, stride=1, padding=1)
        self.lrB = nn.LeakyReLU(0.2)

    def forward(self, x, joinParam):
        y = Functional.interpolate(x, size=[joinParam.size(2), joinParam.size(3)], mode='bilinear', align_corners=True)
        return self.lrB(self.convB(self.lrA(self.convA(torch.cat([y, joinParam], dim=1)))))

class Decoder(nn.Module):
    def __init__(self, featurescnt=1280, dw = .6):
        super(Decoder, self).__init__()
        pms = int(featurescnt * dw)
        self.convol2 = nn.Conv2d(featurescnt, pms, kernel_size=1, stride=1, padding=1)
        self.upspl1 = upsampleBlock(skin=pms//1+320, opFeature=pms//2)
        self.upspl2 = upsampleBlock(skin=pms//2+160, opFeature=pms//4)
        self.upspl3 = upsampleBlock(skin=pms//2+64, opFeature=pms//8)
        self.upspl4 = upsampleBlock(skin=pms//4+32, opFeature=pms//16)
        self.convol3 = nn.Conv2d(pms//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, pms):
        l0, l1, l2, l3, l4 = pms[2], pms[4], pms[6], pms[9], pms[15]
        y0 = self.convol2(l4)
        y1 = self.upspl1(y0, l3)
        y2 = self.upspl2(y1, l2)
        y3 = self.upspl3(y2, l1)
        y4 = self.upspl4(y3, l0)
        return self.convol3(y4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()      
        self.ptmodel = models.densenet169(pretrained=True)

    def forward(self, x):
        pms = [x]
        for _, value in self.ptmodel.pms._modules.items(): pms.append(value(pms[-1]))
        return pms

class UnetMobModel(nn.Module):
    def __init__(self):
        super(UnetMobModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

def custssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    ssim = kornia.losses.SSIM(window_size=11,max_val=val_range)
    return ssim(img1, img2)

class performanceTools(object):
    def __init__(self):
        self.setzero()
    def setzero(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def updateAll(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class loadDf(Dataset):
    os = __import__('os')
    def __init__(self, trainDf, rootDir, transform=None):
        self.trainDf = trainDf
        self.rootDir = rootDir
        self.transform = transform
    def __len__(self):
        return len(self.trainDf)
    def __getitem__(self, i):
        item = self.trainDf[i]
        img_name = os.path.join(self.rootDir,item[0])
        image = (Image.open(img_name))
        dmImgName = os.path.join(self.rootDir,item[1])
        depthMap =(Image.open(dmImgName))
        s={'image': image, 'depth': depthMap}
        if self.transform:  
            s = self.transform({'image': image, 'depth': depthMap})
        return s
    
class dataSynthesis(object):
    def __init__(self, probability):
        self.probability = probability
        self.indices = list(permutations(range(3), 3))
    def __call__(self, ex):
        image, depthMap = ex['image'], ex['depth']
        if not isinstance(image, Image.Image):
            raise TypeError('')
        if not isinstance(depthMap, Image.Image):
            raise TypeError('') 
        #Augmenting data
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depthMap = depthMap.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])    
        return {'image': image, 'depth': depthMap}
    
class toImg(object):
    def __init__(self,test=False):
        self.test = test
    def __call__(self, sample):
        image, depthMap = sample['image'], sample['depth']       
        image = self.toim(image)
        depthMap = depthMap.resize((320, 240))
        if self.test:
            depthMap = self.toim(depthMap).float() / 1000
        else:            
            depthMap = self.toim(depthMap).float() * 1000        
        depthMap = torch.clamp(depthMap, 10, 1000)
        return {'image': image, 'depth': depthMap}

    def toim(self, ip):
        ip = np.array(ip)
        if not ((isinstance(ip, np.ndarray) and (ip.ndim in {2, 3})) or isinstance(ip, Image.Image)):
                raise TypeError('type err2')            
        if isinstance(ip, np.ndarray):
            if ip.ndim==2:
                ip=ip[..., np.newaxis]               
            img = torch.from_numpy(ip.transpose((2, 0, 1)))
            return img.float().div(255)


#loading data
dfTrain = pd.read_csv('data/train.csv')
dfTrain = dfTrain.values.tolist()
dfTrain = shuffle(dfTrain, random_state=2)

#garbage collection
gc.collect()
torch.cuda.empty_cache()

#training steps
unetmodel = UnetMobModel().cuda()
unetmodel = nn.DataParallel(unetmodel)
#model.load_state_dict(torch.load('49.pth'))
ddf = loadDf(trainDf=dfTrain, rootDir='', transform=transforms.Compose([dataSynthesis(0.5),toImg()]))
dl = DataLoader(ddf, 1, shuffle=True)
l1_criterion = nn.L1Loss()
optimizer = torch.optim.Adam(unetmodel.parameters(),0.0001)
for epochs in range(100):
    path=''+str(epochs)+'.pth'        
    torch.save(unetmodel.state_dict(), path)
    bt = performanceTools()
    lossarr = performanceTools()
    unetmodel.train()
    end = time.time()
    for i, batch in enumerate(dl):
        optimizer.zero_grad()
        image = torch.autograd.Variable(batch['image'].cuda())
        depthMap = torch.autograd.Variable(batch['depth'].cuda(non_blocking=True))
        depthNorm = (1000.0/depthMap)
        output = unetmodel(image)
        depthLoss = l1_criterion(output, depthNorm)
        ssimLoss = torch.clamp((1 - custssim(output, depthNorm, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
        loss = (1.0 * ssimLoss.mean().item()) + (0.1 * depthLoss)
        lossarr.updateAll(loss.data.item(), image.size(0))
        loss.backward()
        optimizer.step()
        bt.updateAll(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(bt.val*(len(dl) - i))))
        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            'ETA {eta}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'
            .format(epochs, i, len(dl), loss=lossarr, eta=eta)) 
    savePath='modelEpoch'+str(epochs)+'.pth'        
    torch.save(unetmodel.state_dict(), savePath)
