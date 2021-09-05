import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('stereoip/left/000000_17.png',0)
imgR = cv.imread('stereoip/right/000000_17.png',0)
'''imgL = cv.cvtColor('stereoip/left/000000_00.png', cv.COLOR_BGR2GRAY)
imgR = cv.cvtColor('stereoip/left/000000_00.png', cv.COLOR_BGR2GRAY)'''

blockSize = 11
minDisp = -128
maxDisp = 128
numDisp = maxDisp - minDisp
uniquenessRatio = 5
speckleWindowSize = 200
speckleRange = 2
disp12MaxDiff = 0

sde = cv.StereoSGBM_create(
    minDisparity=minDisp,
    numDisparities=numDisp,
    blockSize=blockSize,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * blockSize * blockSize,
    P2=32 * 1 * blockSize * blockSize,
)
disparity_SGBM = sde.compute(imgL, imgR)
cv.imwrite("stereoip/depth.png", disparity_SGBM)
disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
cv.imshow("Depth", disparity_SGBM)
cv.imwrite("stereoip/depthnormed.png", disparity_SGBM)
'''plt.imshow(disparity_SGBM, cmap='plasma')
plt.savefig("1.png")'''