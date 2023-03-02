# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:05:23 2023

@author: Forest Speed
"""

# Working Conda Environment -> mcSIM_Dev_try


# DMD Stuff -> DLP GUI Must be Closed! 
from mcsim.expt_ctrl import dlp6500
import numpy as np 
from PIL import Image
from skimage import io

# Atlas10 Stuff
from arena_api.system import system
from arena_api.buffer import *
from arena_api.__future__.save import Writer
from arena_api.buffer import BufferFactory
from arena_api import enums
from arena_api.enums import PixelFormat
#from arena_api.__future__.save import Recorder
from arena_api.callback import callback, callback_function


# uManger Stuff
from pycromanager import Core

# MkWIFI Stuff (E-stim...)
import serial 
import csv 
import pandas as pd 
import time 

# Python 1300 Stuff 
from pypylon import pylon
from pypylon import genicam
import sys
import cv2

"""
DAC01 -> Green DMD Trig_1: 0-5V
DAC02 -> Green DMD Trig_2: 0-5V
DAC03 -> Piezo: -10-10V
DAC04 -> Blue DMD Trig_1: 0-5V
DAC05 -> Blue DMD Trig_2: 0-5V
DAC06 -> Green LED: 0-5V
DAC08 -> Blue LED: 0-5V
DAC15 -> Python1300 Input Trig: 0-5V 

Python1300 Pins -> Blue, line3 -> TTL in 
                   Brown, line6 -> Ground

"""

############################################################  ATLAS10 Helper Functions

# Connect to Camera Function 
def create_device_from_serial_number(serial_number):
    camera_found = False
    device_infos = None
    selected_index = None

    device_infos = system.device_infos
    for i in range(len(device_infos)):
        if serial_number == device_infos[i]['serial']:
            selected_index = i
            camera_found = True
            break

    if camera_found == True:
        selected_model = device_infos[selected_index]['model']
        print(f"Create device: {selected_model}...")
        device = system.create_device(device_infos=device_infos[selected_index])[0]
    else:
        raise Exception(f"Serial number {serial_number} cannot be found")
        
    return device


def set_ROI_Atlas10(device, width, height, offset_X, offset_Y):
    nodemap = device.nodemap
    width_node = nodemap['Width']
    while width % width_node.inc:
        width -= 1
    nodemap['Width'].value = width
    height_node = nodemap['Height']
    while height % height_node.inc:
        height -= 1
    nodemap['Height'].value = height
    offset_X_node = nodemap['OffsetX']
    while offset_X % offset_X_node.inc:
        offset_X -= 1
    nodemap['OffsetX'].value = offset_X
    offset_Y_node = nodemap['OffsetY']
    while offset_Y % offset_Y_node.inc:
        offset_Y -= 1
    nodemap['OffsetY'].value = offset_Y


def reset_ROI_Atlas10(device):
    nodemap = device.nodemap
    nodemap['OffsetX'].value = 0
    nodemap['OffsetY'].value = 0
    nodemap['Width'].value = nodemap['Width'].max
    nodemap['Height'].value = nodemap['Height'].max
    
    
def set_Cam_Trigs_Atlas10(device, exposure_Time):
    nodes = device.nodemap.get_node(['TimerSelector', 'TimerTriggerSource', 'TimerTriggerActivation', 'TimerDelay', 'TimerDuration', 'LineSelector', 'LineMode', 'LineSource','LineInverter', 'VoltageExternalEnable' ])
        
    # Setting Line2 to Trigger Output
    nodes['LineSelector'].value = 'Line2'
    nodes['LineMode'].value = 'Output'
    #nodes['LineSource'].value = 'ExposureActive'
    nodes['LineSource'].value = 'Timer0Active'
    nodes['LineInverter'].value = bool(1)
    nodes['TimerSelector'].value = 'Timer0'
    nodes['TimerTriggerSource'].value = 'ExposureStart'
    nodes['TimerTriggerActivation'].value = 'FallingEdge'
    #nodes['TimerDelay'].value = exposure_Time
    nodes['TimerDelay'].value = 0.00
    nodes['TimerDuration'].value = 200.00
    nodes2 = device.nodemap.get_node(['LineSelector', 'VoltageExternalEnable' ])

    # Setting Line4 to V_Supply
    nodes2['LineSelector'].value = 'Line4'
    nodes2['VoltageExternalEnable'].value = bool(1)
    print("Camera Triggers Set!")

# Set defualt stuff for ATLAS10
def set_Cam_Defaults_Atlas10(device):
    nodemap = device.nodemap
    nodes = nodemap.get_node(['ExposureAuto', 'ExposureTime'])
    exposure_auto_initial = nodes['ExposureAuto'].value
    exposure_time_initial = nodes['ExposureTime'].value
    print(exposure_auto_initial, exposure_time_initial)

    # Setup Stream
    tl_stream_nodemap = device.tl_stream_nodemap
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True
    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    #tl_stream_nodemap["StreamBufferHandlingMode"].value = "OldestFirst"
    new_pixel_format = 'Mono8'
    nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
    #nodes['PixelFormat'].value = new_pixel_format
    return new_pixel_format, exposure_time_initial #Removed Device from Returns

# Lock Atlas Exposure
def lock_Exposure_Atlas10(device):
    nodemap = device.nodemap
    nodes = nodemap.get_node(['ExposureAuto', 'ExposureTime'])
    nodes['ExposureAuto'].value = "Off"
    
# Put Atlas10 in Multi-image mode
def set_Mode_Multi_Atlas10(device):
    nodemap = device.nodemap
    nodes = nodemap.get_node(['AcquisitionMode','AcquisitionFrameCount'])
    nodes['AcquisitionMode'].value = "MultiFrame"
    nodes['AcquisitionFrameCount'].value = 3

# Put Atlas10 in Continuous mode
def set_Mode_Cont_Atlas10(device):
    nodemap = device.nodemap
    nodes = nodemap.get_node(['AcquisitionMode'])
    nodes['AcquisitionMode'].value = "Continuous"

    
    
# TODO -> Add MultiThread Stuff !!!!!!



###################################################### DMD Helper Functions

# Upload Mask to DMD
def Upload_Arr_to_DMD(arr,dmd):
    pat = arr
    exposure_t = 150
    dark_t = 0
    triggered = False
    img_inds, bit_inds = dmd.upload_pattern_sequence(pat, exposure_t, dark_t, triggered, clear_pattern_after_trigger = False, bit_depth = 1, num_repeats = 0)
    dmd.set_pattern_sequence(img_inds, bit_inds, exposure_t, dark_t, triggered=False, clear_pattern_after_trigger= False, bit_depth=1, num_repeats=0, mode='on-the-fly')
    dmd.start_stop_sequence("start")
    
# Upload Circle to DMD
def Upload_FB_to_DMD(dmd):
    img = np.array(Image.open('patterns//FB.bmp'))
    img1 = img/247
    img11 = img1.astype(np.uint8)
    Upload_Arr_to_DMD(img11, dmd)
    
def Upload_SIM1_to_DMD(dmd):
     img = np.array(Image.open('patterns//SIM_Pat_11_bmp.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)  

def Upload_SIM2_to_DMD(dmd):
     img = np.array(Image.open('patterns//SIM_Pat_22_bmp.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)  
     
     
     
def Upload_SIM3_to_DMD(dmd):
     img = np.array(Image.open('patterns//SIM_Pat_33_bmp.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)      

    


def set_DMD_SIM(exposure_time_cam):
    
    # import tif file into numpy array 
    pats = io.imread('patterns//OS_NEIL_PATS_Extended.bmp')                                         #!!!!!!!! EXTRA for Z slices! 
    #pats[np.where(pats!=0)] = 1
    # Get DMD & Upload Pats
    dmd = dlp6500.dlp6500win(debug=False, index = 1)
    exposure_t = int(exposure_time_cam - 250)
    dark_t = 0
    triggered = True
    img_inds, bit_inds = dmd.upload_pattern_sequence(pats, exposure_t, 0, triggered, clear_pattern_after_trigger = True, bit_depth = 1, num_repeats = 0)
    print("Upload Complete")
    dmd.set_pattern_sequence(img_inds, bit_inds, exposure_t, dark_t, triggered, clear_pattern_after_trigger= True, bit_depth=1, num_repeats=0, mode='on-the-fly')
    dmd.start_stop_sequence("start")
    print("DMD IS INITIALIZED")
    return dmd


###################################################### TriggerScope Helpers



def set_MM_DMD_Trigs():
    mmc = Core()

    #Trig1
    mmc.set_property("TS_DAC01", "Blanking","On") 
    mmc.set_property("TS_DAC01", "State","1")
    mmc.set_property("TS_DAC01", "Volts", str(3.3))

    #Trig2
    mmc.set_property("TS_DAC02", "State","1")
    mmc.set_property("TS_DAC02", "Volts", str(3.3))


def turn_Off__MM_Trigs():
    mmc = Core()
    mmc.set_property("TS_DAC01", "State","0")
    mmc.set_property("TS_DAC02", "State","0")









###################################################### Python1300 Helpers
def gui_pylon():
    try:
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(1)
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())

        # Start the grabbing of c_countOfImagesToGrab images.
        # The camera device is parameterized with a default configuration which
        # sets up free-running continuous acquisition.
        camera.StartGrabbingMax(10000, pylon.GrabStrategy_LatestImageOnly)

        while camera.IsGrabbing():
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                imageWindow.SetImage(grabResult)
                imageWindow.Show()
            else:
                print("Error: ",
                  grabResult.ErrorCode)  # grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError
            grabResult.Release()
            time.sleep(0.05)

            if not imageWindow.IsVisible():
                camera.StopGrabbing()

        # camera has to be closed manually
        camera.Close()
        # imageWindow has to be closed manually
        imageWindow.Close()

    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e.GetDescription())






    
# Set Python1300 to follow Atlas10 -> Still need to figure out GPIO connection
def set_Python1300_Slave(camera):
    camera.MaxNumBuffer = 1
    camera.TriggerSource ='Line3'
    camera.TriggerSelector ='FrameStart'
    camera.TriggerMode ='On'
    camera.TriggerActivation='FallingEdge'
    #camera.AcquisitionMode='Continuous'
    print("Python1300 slave mode: ", camera.TriggerMode.Value)




########################################################## Main Script
#gui_pylon()

"""
# Connect to Atlas10
Atlas10 = create_device_from_serial_number("220600074")

# Connect to Python1300
Python1300 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
Python1300.Open()

set_Python1300_Slave(Python1300)

"""
# Connect to DMDs
dmd1 = dlp6500.dlp6500win(debug=False, dmd_index = 0) #Blue
dmd2 = dlp6500.dlp6500win(debug=False, dmd_index = 1) #Green




#dmd1 = dlp6500.dlp6500win(debug=False)  
#dmd2 = dlp6500.dlp6500win2(debug=False) 

# Connect to uManager/TriggerScope
#core = Core()

# Connect to Arduino for Electrical Recording..
#arduino = serial.Serial('COM8', 115200, timeout=.1)





Upload_FB_to_DMD(dmd1)
Upload_FB_to_DMD(dmd2)
#Upload_SIM3_to_DMD(dmd1)
#Upload_SIM3_to_DMD(dmd2)








