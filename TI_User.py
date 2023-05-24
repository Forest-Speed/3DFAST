# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:46:48 2023

@author: FS

TI (Basic)
"""

import hardware_control_2023 as HC
import numpy as np
from mcsim.analysis import fit_dmd_affine, mm_io
from localize_psf import affine, rois, fit
from pathlib import Path
import json

window_width = 2000
window_height = 1300
DMD_Width = 1920
DMD_Height = 1080
Camera_Width = 2448 
Camera_Height = 2048 


blue_Cam = HC.create_device_from_serial_number("220600074")
#red_Cam = HC.create_device_from_serial_number("224500525") 
blue_DMD = HC.dlp6500.dlp6500win(debug=False, dmd_index = 0)
#red_DMD = HC.dlp6500.dlp6500win(debug=False, dmd_index = 1) 


# Upload Full Circle to Both DMDs
HC.Upload_FB_to_DMD(blue_DMD)
#HC.Upload_FB_to_DMD(red_DMD)  

# Save Current Camera Settings
HC.save_UserSet1_Atlas10(blue_Cam)
#HC.save_UserSet2_Atlas10(red_Cam)

# Reset Camera Settings to defaults (for DMD/cam pixel calibration)
HC.reset_Atlas10_Default(blue_Cam)
HC.set_Gain_Auto(blue_Cam)
#HC.reset_Atlas10_Default(red_Cam)


num_Targets = 2 # Change Each Trial!!!!
ix_Blue = np.ones(num_Targets)
iy_Blue = np.ones(num_Targets)
fx_Blue = np.ones(num_Targets)
fy_Blue = np.ones(num_Targets)



for i in range(num_Targets):
    # Snap&Draw -> returns coordinates of box drawn
    ix_Blue[i], iy_Blue[i], fx_Blue[i], fy_Blue[i] = HC.snap_Image(blue_Cam) # Escape Key To Exit!
    #ix_Red, iy_Red, fx_Red, fy_Red = HC.snap_Image(red_Cam) # Escape Key To Exit!




# Making Camera Masks
camera_Mask_Blue_Full = np.zeros((Camera_Height, Camera_Width), np.uint8)

for k in range(num_Targets):
    camera_Mask_Blue = np.ones((Camera_Height, Camera_Width), np.uint8)
    
    for i in range(Camera_Width):
        if i < ix_Blue[k]:
            camera_Mask_Blue[:,i] = 0
        elif i > fx_Blue[k]:
            camera_Mask_Blue[:,i] = 0
        #if i < ix_Red:
        #    camera_Mask_Red[:,i] = 0
        #elif i > fx_Red:
        #    camera_Mask_Red[:,i] = 0            

    for j in range(Camera_Height):
        if j < iy_Blue[k]:
            camera_Mask_Blue[j,:] = 0
        elif j > fy_Blue[k]:
            camera_Mask_Blue[j,:] = 0
        #if j < iy_Red:
        #     camera_Mask_Red[j,:] = 0
        #elif j > fy_Red:
        #     camera_Mask_Red[j,:] = 0  

    camera_Mask_Blue_Full = camera_Mask_Blue_Full + camera_Mask_Blue 

for i in range(Camera_Height):
    for j in range(Camera_Width):
        if camera_Mask_Blue_Full[i,j] > 1:
            camera_Mask_Blue_Full[i,j] = 1
            
        

# Get Transform saved with calibration - BLUE!  -> NEED TO PUT CORRECT PATH
img_fname = Path("C:/Users/Researcher/mcSIM/examples/data", "calibrate_5_22.tiff")
save_dir = img_fname.parent /"2023_05_22_13;31;16_affine_calibration"



fname_sum = save_dir / "affine_xform_blue.json"
with open(fname_sum) as f:
   dmd_affine_transformations = json.load(f)

affine_xform_list = list(dmd_affine_transformations.values())
affine_xform_values = affine_xform_list[0]
affine_xform = np.ones((3,3))
affine_xform[0] = affine_xform_values[0]
affine_xform[1] = affine_xform_values[1]
affine_xform[2] = affine_xform_values[2]
affine_xform_inv = np.linalg.inv(affine_xform)
dmd_all_on = np.ones((DMD_Height, DMD_Width),np.uint8)
dmd_coords = np.meshgrid(range(dmd_all_on.shape[1]), range(dmd_all_on.shape[0]))

# Make Blue DMD Mask
dmd_Pat_Blue = affine.xform_mat(camera_Mask_Blue_Full, affine_xform_inv, dmd_coords, mode='nearest')
dmd_Pat_Blue[np.isnan(dmd_Pat_Blue)] = 0
dmd_Pat_Blue = dmd_Pat_Blue.astype('uint8')

HC.Upload_Arr_to_DMD(dmd_Pat_Blue,blue_DMD)












"""
# Get Transform saved with calibration - Red!
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
dmd_all_on = np.ones((DMD_Height, DMD_Width),np.uint8)
dmd_coords = np.meshgrid(range(dmd_all_on.shape[1]), range(dmd_all_on.shape[0]))

# Make Blue DMD Mask
dmd_Pat_Red = affine.xform_mat(camera_Mask_Blue, affine_xform_inv, dmd_coords, mode='nearest')
dmd_Pat_Red[np.isnan(dmd_Pat_Red)] = 0
dmd_Pat_Red = dmd_Pat_Red.astype('uint8')
"""








# Reload Original Cam Settings
HC.reload_UserSet1_Atlas10(blue_Cam)
#HC.reload_UserSet2_Atlas10(red_Cam)

