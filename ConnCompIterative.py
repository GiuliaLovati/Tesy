from structure_tensor import *
from ConnectedComp import *

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import figure

def totaldenoising(namelist):
    newimgs=[]
    for name in namelist:
        imgDenoised = ApplyDenoising(name)
        newimgs.append(imgDenoised)
    return newimgs


#interactive cropping

def Cropping(image):
    imshow(image)
    plt.pause(0.05)
    print('Image original dimensions:',image.shape)
    xin=int(image.shape[1]-image.shape[1]*3/4)       #(TO IMPROVE)
    yin=int(image.shape[0]-image.shape[0]*3/4)
    xstep=int(image.shape[1]/2)
    ystep=int(image.shape[0]/2)
    imgInTiny = image[yin:yin+ystep,xin:xin+xstep]
    imshow(imgInTiny)
    plt.pause(0.05)
    dim = input('Cropped well? Answer y/n: ')
    while dim == 'n':
        yin = int(input('Insert new y_in [px]:'))
        xin = int(input('Insert new x_in [px]:'))
        ystep = int(input ('Insert new y_widht [px]:'))
        xstep = int(input ('Insert new x_widht [px]:'))
        imgInTiny = image[yin:yin+ystep,xin:xin+xstep]
        imshow(imgInTiny)
        plt.pause(0.05)
        dim = input('Cropped well? Answer y/n: ')
    return imgInTiny

def iterativeCropping(imglist):
    for i in range(len(imglist)):
        imglist[i] = Cropping(imglist[i])
    return imglist 


#threshold value as mean + 1 standard deviation

from scipy.stats import norm

def thresholdvalue(imglist):
    means = []
    stds = []
    thr = []
    for i in range(len(imglist)):
        param = norm.fit(imglist[i]) 
        mean = param[0]
        means.append(mean)
        sd = param[1]
        stds.append(sd)
        thr.append(mean +sd)
    return means, stds, thr


#Connected Component selection
def iterativeConnComp(imglist):
    index = []
    newimg = []
    huenewimg = []
    means, stds, thr = thresholdvalue(imglist)
    for i in range(len(imglist)):
        index.append(selection(imglist[i], thr[i])[0])    #using as a threshold mean + 1 standard deviation 
        newimg.append(newimgbigcomponents(imglist[i],index[i], thr[i])[0])
        huenewimg.append(newimgbigcomponents(imglist[i],index[i], thr[i])[1])
        print ('threshold:', thr[i])
        figure(figsize=(20,10))
        imshow(newimg[i])
        plt.pause(0.05)
        dim = input('Selected well? Answer y/n: ')
        while dim == 'n':
            newthr = float(input('Insert new threshold value:'))
            number = float(input('Number of time you want to subtract (or add with sign -) the std to mean area value:'))
            index[i] = differentSelection(imglist[i], int(newthr), number)[0]  #(TO IMPROVE NUMBER PART)
            newimg[i] = newimgbigcomponents(imglist[i], index[i], newthr)[0]
            huenewimg[i] = newimgbigcomponents(imglist[i],index[i], newthr)[1]
            figure(figsize=(20,10))
            imshow(newimg[i])
            plt.pause(0.05)
            dim = input('Selected well? Answer y/n: ')
    return newimg , huenewimg


def iterativeEdges(newimg, huenewimg):  
    edgesit = []     #np.zeros_like(newimg)
    upperlimitxit = []
    upperlimityit = []
    lowerlimitxit = []
    lowerlimityit = []
    for i in range(len(newimg)):
        edges, upperlimitx, upperlimity = FindingUpperEdges(newimg[i], huenewimg[i])
        edges, lowerlimitx, lowerlimity = FindingLowerEdges(newimg[i], huenewimg[i], edges)
        figure(figsize=(20,10))
        imshow(edges)
        plt.pause(0.05)
        edgesit.append(edges)
        upperlimitxit.append(upperlimitx)
        upperlimityit.append(upperlimity)
        lowerlimitxit.append(lowerlimitx)
        lowerlimityit.append(lowerlimity)
    return edgesit, upperlimitxit, upperlimityit, lowerlimitxit, lowerlimityit  


#Nice images to explain
def ImgWithEdges(imglist, upx, upy, lox, loy):
    for i in range(len(imglist)):
        figure(figsize=(20,10))
        imshow(imglist[i])
        plt.scatter(upy[i],upx[i], c='k', s=1)
        plt.scatter(loy[i],lox[i], c='k', s=1)
        plt.pause(0.05)
        #plt.savefig('tshirtPT.png', dpi=1600)
        
        
def ThicknessIt(upperlimitx, upperlimity, lowerlimitx, lowerlimity):
    deltacolumnit = []
    deltait = []
    deltasecit = []
    deltakmit = [] 
    deltamit = []
    for i in range(len(upperlimitx)):
        print ('IMAGE', i)
        deltacolumn, delta = Thickness(upperlimity[i], upperlimitx[i], lowerlimity[i], lowerlimitx[i])
        #print(deltacolumn)
        #print(delta)
        #plt.plot(deltacolumn, delta)
        #plt.xlabel('Image Column')
        #plt.ylabel('Connected Component Thickness [px]')
        plt.pause(0.05)
        deltacolumnit.append(deltacolumn)
        deltait.append(delta)
        deltasec, deltakm, deltam = Conversion(delta)
        deltasecit.append(deltasec)
        deltakmit.append(deltakm)
        deltamit.append(deltam)
        plt.plot(deltacolumn, deltam)
        plt.xlabel('Image Column')
        plt.ylabel('Connected Component Thickness [m]')
        plt.pause(0.05)
    return deltacolumnit, deltait, deltasecit, deltakmit, deltamit