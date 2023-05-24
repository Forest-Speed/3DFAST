# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:58:57 2023

@author: Researcher
"""

import numpy as np 
import matplotlib.pyplot as plt
import ANALYZE_3DFAST


spikes_estimate = np.load(r'C:\users\researcher\3DFAST-main\3DFAST-main\3DFAST DATA\Sean Processed Data\5-22\Fusion\volpy_SV14_pAce_052223_Rec1_HA_adaptive_threshold.npy', allow_pickle = 'true').item()


fs = 600 # CHANGE THIS BASED ON FRAME RATE
ROI_num = 1 #CHange based on Mask-RCNN
num_frames = 12351 #Change BASED ON NUM FRAMES


ROIs = spikes_estimate['ROIs']
spikes = spikes_estimate['spikes']
dFF = spikes_estimate['dFF']
subs = spikes_estimate['t_sub']

dFF_1 = dFF[ROI_num-1]
spikes_1 = spikes[ROI_num-1]

x_Vals,template = ANALYZE_3DFAST.spike_Template(dFF_1,spikes_1,600)
plt.plot(x_Vals,template)




    