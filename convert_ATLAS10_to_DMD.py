# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 08:18:46 2022

@author: Researcher
"""

import datetime
import numpy as np
from pathlib import Path
import json
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mcsim.analysis import fit_dmd_affine, mm_io
#import mcsim.analysis.dmd_patterns as dmd
from matplotlib.patches import Rectangle
from localize_psf import affine, rois, fit
import cv2

from mcsim.expt_ctrl import dlp6500

global start_Pos, stop_Pos, flag, ix, iy, drawing, display_img, fx, fy

ix = 0
iy = 0
fx = 2448
fy = 2048
drawing = False
window_width = 2000
window_height = 1300

TAB1 = "  "
TAB2 = "    "
DMD_Width = 1920
DMD_Height = 1080
Camera_Width = 2448 
Camera_Height = 2048 

# Get Transform saved with calibration - DO THIS EVERY TIME BEFORE AQUISITION W/ SLIDE!
img_fname = Path("data", "test_calib_1220.tiff")
save_dir = img_fname.parent /"2022_12_20_14;58;57_affine_calibration"
fname_sum = save_dir / "affine_xform_red.json"
with open(fname_sum) as f:
   dmd_affine_transformations = json.load(f)

affine_xform_list = list(dmd_affine_transformations.values())
affine_xform_values = affine_xform_list[0]

affine_xform = np.ones((3,3))
affine_xform[0] = affine_xform_values[0]
affine_xform[1] = affine_xform_values[1]
affine_xform[2] = affine_xform_values[2]

affine_xform_inv = np.linalg.inv(affine_xform)

# Make Image 
img = np.ones((Camera_Height, Camera_Width),np.uint8)
img_coords = np.meshgrid(range(img.shape[1]), range(img.shape[0]))



dmd_all_on = np.ones((DMD_Height, DMD_Width),np.uint8)
dmd_coords = np.meshgrid(range(dmd_all_on.shape[1]), range(dmd_all_on.shape[0]))

# Convert to DMD Space
dmd_pattern = affine.xform_mat(img, affine_xform_inv, dmd_coords, mode='nearest')
dmd_pattern[np.isnan(dmd_pattern)] = 0
dmd_pattern = dmd_pattern.astype('uint8')

dmd = dlp6500.dlp6500win(debug=False) 

# DMD Upoload Function
def Upload_Arr_to_DMD(arr,dmd):
    pat = arr
    exposure_t = 150
    dark_t = 0
    triggered = False
    img_inds, bit_inds = dmd.upload_pattern_sequence(pat, exposure_t, dark_t, triggered, clear_pattern_after_trigger = False, bit_depth = 1, num_repeats = 0)
    dmd.set_pattern_sequence(img_inds, bit_inds, exposure_t, dark_t, triggered=False, clear_pattern_after_trigger= False, bit_depth=1, num_repeats=0, mode='on-the-fly')
    dmd.start_stop_sequence("start")
    

def draw_reactangle_with_drag(event, x, y, flags, param):
    
    # ix is starting x coord, it is easiest to start left and move right 
    # iy is the starting y coord, this goes top to bottem 
    # fx is the final x coord, this should be the right most 
    # fy is the final y coord, this should toward bottem 
    
    global ix, iy, drawing, display_img, fx, fy
    #print("Draw Time")
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
        print(iy)
       
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img2 = cv2.imread("12_20_test.jpg") # Change this to full FOV frame taken of real sample!!!
            cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(255,255,255),thickness=10)
            display_img = img2

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img2 = cv2.imread("12_20_test.jpg") # Change this to full FOV frame taken of real sample!!!
        cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(255,255,255),thickness=10)
        display_img = img2
        fx = x
        fy = y  


test_path = "12_20_test.jpg" # Change this to full FOV frame taken of real sample!!!
key = -1
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", draw_reactangle_with_drag)
display_img = cv2.imread(test_path)
cv2.resizeWindow("Image", window_width, window_height)

# Display Until Esc key hit!
while True:
    cv2.imshow("Image", display_img)
    if cv2.waitKey(10) == 27:
        break

print("The box drawn has corners (cam dimensions) at: ",ix,iy,fx,fy)
cv2.destroyAllWindows()

# Use box to make MASK
camera_mask = np.ones((Camera_Height, Camera_Width), np.uint8)

for i in range(Camera_Width):
        if i < ix:
            camera_mask[:,i] = 0
        elif i > fx:
            camera_mask[:,i] = 0

for j in range(Camera_Height):
        if j < iy:
            camera_mask[j,:] = 0
        elif j > fy:
            camera_mask[j,:] = 0
            
            

dmd_pattern = affine.xform_mat(camera_mask, affine_xform_inv, dmd_coords, mode='nearest')
dmd_pattern[np.isnan(dmd_pattern)] = 0
dmd_pattern = dmd_pattern.astype('uint8')
Upload_Arr_to_DMD(dmd_pattern,dmd)








































