# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:05:34 2022

@author: Forest Speed
"""
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

arr1 = np.zeros((1080,1920),np.uint8)
arr2 = np.zeros((1080,1920),np.uint8)
arr3 = np.zeros((1080,1920),np.uint8)



line_Width = 18 


# Traditional OS Pats
j = 1
while j < 1891 :
    i = 0 
    while i < 18 :
        arr1[:,i+j] = 1
        arr2[:,i+j+18] = 1
        arr3[:,i+j+36] = 1
        i = i + 1 
    j = j + 54
    
   
h = 1080
w = 1920
center = (960,540) 
radius = 336   
Y,X = np.ogrid[:h,:w]   
dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
mask = dist_from_center <= radius

mask_Arr = np.ones((1080,1920),np.uint8)
mask_Arr[~mask]=0

arr1 = arr1*mask_Arr*256
arr2 = arr2*mask_Arr*256
arr3 = arr3*mask_Arr*256

print(arr1)



"""
im1 = Image.fromarray(arr1)
im1.save("SIM_Pat_1.tif")

im2 = Image.fromarray(arr2)
im2.save("SIM_Pat_2.tif")

im3 = Image.fromarray(arr3)
im3.save("SIM_Pat_3.tif")
"""
