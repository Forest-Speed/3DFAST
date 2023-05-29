# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:21:00 2023

@author: FS
"""
import numpy as np 


    

def load_Data(fname,ROI_Num):
    
    #TODO ADD SNR
    
    data = np.load(fname, allow_pickle = 'true').item()
    spikes_All =data['spikes']
    dFF_All = data['dFF']
    subs_All = data['t_sub']
    dFF= dFF_All[ROI_Num]
    spikes_Estimate = spikes_All[ROI_Num]
    subs = subs_All[ROI_Num]
    
    return  dFF, spikes_Estimate, subs


def spike_Template(dFF,spike_times,Fs):
    
    #TODO, Deal with rapid spiking
    # CASE1 - no spikes 20 timepoints before/after
    # Case2 there are spikes within the plotting window
    #   -REMOVE BEFORE AVERGAING
    
    
    template = np.zeros(41)
    num_Useable_Spikes = 0 
    for spike_time in spike_times:
        if spike_time > 20:
            num_Useable_Spikes = num_Useable_Spikes + 1
            dFF_array = dFF[spike_time-20:spike_time+21]
            template = template + dFF_array
            
            # make dummy dFF
            
            
            
    template = template/num_Useable_Spikes
    x_Vals = np.linspace(0,40,41)/Fs
    return x_Vals, template


def visualize_Data(dFF, spike_Times, Fs, subs):
    # Plot Spike template 
    
    
    # Plot dFF w/ spikes
    
    # Plot PSTH with stimulation times 
    
    
    
    
    
