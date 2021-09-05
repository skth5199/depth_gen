from tkinter import *
import tkinter as tk
from tkinter import font  as tkfont
from tkinter import filedialog
from PIL import Image
import os
import numpy as np
import torch
import cv2
import gc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
import torchvision.models as models
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import torch.nn.functional as Functional

class upsampleBlock(nn.Sequential):
    def __init__(self, skin, opFeature):
        super(upsampleBlock, self).__init__()        
        self.convA = nn.Conv2d(skin, opFeature, kernel_size=3, stride=1, padding=1)
        self.lrA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(opFeature, opFeature, kernel_size=3, stride=1, padding=1)
        self.lrB = nn.LeakyReLU(0.2)

    def forward(self, x, joinParam):
        y = Functional.interpolate(x, size=[joinParam.size(2), joinParam.size(3)], mode='bilinear', align_corners=True)
        return self.lrB( self.convB( self.lrA(self.convA( torch.cat([y, joinParam], dim=1) ) ) )  )

class Decoder(nn.Module):
    def __init__(self, featurescnt=1280, dw = .6):
        super(Decoder, self).__init__()
        pms = int(featurescnt * dw)
        self.conv2 = nn.Conv2d(featurescnt, pms, kernel_size=1, stride=1, padding=1)
        self.up0 = upsampleBlock(skin=pms//1 + 320, opFeature=pms//2)
        self.up1 = upsampleBlock(skin=pms//2 + 160, opFeature=pms//2)
        self.up2 = upsampleBlock(skin=pms//2 + 64, opFeature=pms//4)
        self.up3 = upsampleBlock(skin=pms//4 + 32, opFeature=pms//8)
        self.up4 = upsampleBlock(skin=pms//8 +  24, opFeature=pms//8)
        self.up5 = upsampleBlock(skin=pms//8 +  16, opFeature=pms//16)
        self.conv3 = nn.Conv2d(pms//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, pms):
        l0, l1, l2, l3, l4, l5, l6 = pms[2], pms[4], pms[6], pms[9], pms[15], pms[18], pms[19]
        y0 = self.conv2(l6)
        y1 = self.up0(y0, l5)
        y2 = self.up1(y1, l4)
        y3 = self.up2(y2, l3)
        y4 = self.up3(y3, l2)
        y5 = self.up4(y4, l1)
        y6 = self.up5(y5, l0)
        return self.conv3(y6)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()      
        self.original_model = models.mobilenet_v2( pretrained=True )

    def forward(self, inp):
        features = [inp]
        for _, value in self.original_model.features._modules.items(): features.append( value(features[-1]) )
        return features

class UnetMobModel(nn.Module):
    def __init__(self):
        super(UnetMobModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class loadDf(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,os.listdir(self.root_dir)[idx])
        i = (Image.open(img_name))
        s={'image': i}
        if self.transform:  
            s = self.transform({'image': i})
        return s

class toImg(object):
    def __init__(self,test=False):
        self.test = test

    def __call__(self, s):
        img = s['image']
        img = img.resize((640, 480))
        img = self.toimg(img)
        return {'image': img}

    def toimg(self, ip):
        ip = np.array(ip)
        if not ((isinstance(ip, np.ndarray) and (ip.ndim in {2, 3})) or isinstance(ip, Image.Image)):
                raise TypeError('type err')
                             
        if isinstance(ip, np.ndarray):
            if ip.ndim==2:
                ip=ip[..., np.newaxis]
            i = torch.from_numpy(ip.transpose((2, 0, 1)))
            return i.float().div(255)

class DepthApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Helvetica', size=48, weight="bold")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames={}
        for F in (HomePage, DeEst):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("HomePage")
        
    def show_frame(self,page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#ffffff')
        self.controller = controller
        title = Label(self, text="\nDepth Estimation", font=('ms serif',36,"bold"), bg="#ffffff")
        title.pack()
        subtitle = Label(self, text="Generation of Depth maps using Deep Learning", font=('courier new',16), bg="#ffffff")
        subtitle.pack()     
        enter = Button(self, text='Enter', font=('courier new', 26),
                              bg='black',
                              fg='white', padx=5, pady=5, width=15,
                              command=lambda: controller.show_frame("DeEst"))
        enter.place(x=290,y=550)
     
class DeEst(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#ffffff') 
        self.controller = controller
        self.folder=None
        self.x = 111
        def load_folder():
            self.folder = filedialog.askdirectory()
            print(self.folder+" has been loaded")
            load_info["text"]="Folder has been loaded"
            
        def depthEstimator():
            dirs = os.listdir(self.folder)
            print(self.folder)
            imgdir = self.folder
            if v.get() == 0:
                ddf = loadDf(root_dir=imgdir)
                fig = plt.figure()
                len(ddf)
                for i in range(len(ddf)):
                    sample = ddf[i]
                    print(i, sample['image'].size)
                    plt.imshow(sample['image'])
                    plt.figure()
                    if i == 6:
                        plt.show()
                        break
                ddf = loadDf(root_dir=imgdir,transform=transforms.Compose([toImg()]))
                batch_size=1
                tdl=torch.utils.data.DataLoader(ddf, batch_size)
                dataiter = iter(tdl)
                images = dataiter.next()
                gc.collect()
                torch.cuda.empty_cache()
                unetmodel = UnetMobModel().cuda() 
                unetmodel.load_state_dict(torch.load('TrainedUnetMobilenetv2.pth'),strict=False)
                unetmodel.eval()
                gc.collect()
                torch.cuda.empty_cache()
                for i,sample_batched1  in enumerate (tdl):
                    image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
                    outtt=unetmodel(image1 )
                    x=outtt.detach().cpu().numpy()
                    img=x.reshape(240,320)
                    scale_percent = 200 
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    #change the name of the output folder here accordingly
                    plt.imsave('output/%dc.jpg' %i, resized, cmap='gray_r')
            else:
                #os.system('python C:/Users/srika/Desktop/Pix2PixDepthEst/test.py')
                #change name of image as needed
                src_image = cv2.imread('input/1.jpg')
                src_image = img_to_array(src_image)
                src_image = (src_image - 127.5) / 127.5
                pixels = np.expand_dims(src_image, axis=0)
                print(pixels.shape)
                model = load_model('pix.h5')
                gen_image = model.predict(pixels)
                plt.imsave('output/op.jpg', cmap='inferno')
            class_info["text"]="Conversion Complete"

        title = Label(self, text="\nDepth map generator", font=('courier new',24,"bold"), bg="#ffffff")
        title.pack()
        load_txt = Label(self, text="Load Images", font=('courier new',16,"bold"), bg="#ffffff")
        load_txt.place(x=60, y=100)
        load_select = Button(self, text="Choose Folder", font=('courier new',10,"bold"), width=50, command=load_folder)
        load_select.place(x=60, y=150)
        
        #Folder loading
        load_info = Label(self, text="No Folder", font=('courier new', 10), bg="#ffffff")
        load_info.place(x=100, y=200)

        rb1 = Label(self, text="", font=('courier new', 10), bg="#ffffff")
        rb1.place(x=650, y=120)
        rb2 = Label(self, text="", font=('courier new', 10), bg="#ffffff")
        rb2.place(x=650, y=150)

        v = tk.IntVar()   
        tk.Radiobutton(rb1,
               text="Indoor",
               padx = 20, 
               variable=v, 
               value=0).pack(anchor=tk.W)
        tk.Radiobutton(rb2,
               text="Outdoor",
               padx = 20, 
               variable=v, 
               value=1).pack(anchor=tk.W)

        # Detect depth
        detect_txt = Label(self, text="Generate depth maps", font=('courier new',16,"bold"), bg="#ffffff")
        detect_txt.place(x=60, y=250)
        detect_select = Button(self, text="Start Processing", font=('courier new',10,"bold"), width=50, command=depthEstimator)
        detect_select.place(x=60, y=300)
        
        # Classification Status
        class_info = Label(self, text="Computing...", font=('courier new', 10), bg="#ffffff")
        class_info.place(x=100, y=350)

        output_file = Label(self, text="Output", font=('courier new',16,"bold"), bg="#ffffff")
        output_file.place(x=60, y=400)
        output_info = Label(self, text="Output Location: Inside the Loaded Folder", font=('courier new', 10), bg="#ffffff")
        output_info.place(x=100, y=500)

if __name__ == "__main__":
    app = DepthApp()
    app.geometry("900x650")
    app.resizable(width=False, height=False)
    app.configure(background='white')
    app.mainloop()
 