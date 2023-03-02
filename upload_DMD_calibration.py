# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:03:40 2022

@author: Researcher
"""

import datetime
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mcsim.analysis import fit_dmd_affine, mm_io
import mcsim.analysis.dmd_patterns as dmd_pats

from mcsim.expt_ctrl import dlp6500






dmd_size = [1920, 1080]
masks, radii, pattern_centers = dmd_pats.get_affine_fit_pattern(dmd_size)
mask = masks[1]
mask = mask.astype('uint8')

# Connect to DMD
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
    
    
    
    
mask_Arr = np.ones((1080,1920), np.uint8)
mask_Arr[~mask]=0
Upload_Arr_to_DMD(mask,dmd)