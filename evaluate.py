import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
gt = cv2.imread('genimg/1.jpg', cv2.IMREAD_GRAYSCALE)
gn = cv2.imread('genimg/990.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('23232323.png',gt)
print('1')
print(mse(gt,gn))
print(ssim(gt,gn,multichannel=True))

gt = cv2.imread('genimg/2.jpg',cv2.IMREAD_GRAYSCALE)
gn = cv2.imread('genimg/991.jpg', cv2.IMREAD_GRAYSCALE)
print('2')
print(mse(gt,gn))
print(ssim(gt,gn,multichannel=True))

gt = cv2.imread('genimg/3.jpg',cv2.IMREAD_GRAYSCALE)
gn = cv2.imread('genimg/992.jpg',cv2.IMREAD_GRAYSCALE)
print('3')
print(mse(gt,gn))
print(ssim(gt,gn,multichannel=True))

gt = cv2.imread('genimg/4.jpg',cv2.IMREAD_GRAYSCALE)
gn = cv2.imread('genimg/993.jpg',cv2.IMREAD_GRAYSCALE)
print('4')
print(mse(gt,gn))
print(ssim(gt,gn,multichannel=True))

gt = cv2.imread('genimg/5.jpg',cv2.IMREAD_GRAYSCALE)
gn = cv2.imread('genimg/994.jpg',cv2.IMREAD_GRAYSCALE)
print('5')
print(mse(gt,gn))
print(ssim(gt,gn,multichannel=True))

gt = cv2.imread('genimg/6.png',cv2.IMREAD_GRAYSCALE)
gn = cv2.imread('genimg/995.jpg',cv2.IMREAD_GRAYSCALE)
print('6')
print(mse(gt,gn))
print(ssim(gt,gn,multichannel=True))
