# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:21:00 2023

@author: FS
"""
import numpy as np 


def spike_Template(dFF,spike_times,Fs):
    template = np.zeros(41)
    num_Useable_Spikes = 0 
    for spike_time in spike_times:
        if spike_time > 20:
            num_Useable_Spikes = num_Useable_Spikes + 1
            dFF_array = dFF[spike_time-20:spike_time+21]
            template = template + dFF_array
    template = template/num_Useable_Spikes
    x_Vals = np.linspace(0,40,41)/Fs
    return x_Vals, template
    
    
    