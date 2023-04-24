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
from matplotlib import pyplot as plt

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
      -> Blue LED: 0-5V
DAC08 -> Python1300 Input Trig: 0-5V 

Python1300 Pins -> Blue, line3 -> TTL in 
                   Brown, line6 -> Ground

"""

global start_Pos, stop_Pos, flag,ix, iy, drawing, display_img, fx, fy, DMD_Height, DMD_Width

ix = 0
iy = 0
fx = 2448
fy = 2048
drawing = False
DMD_Width = 1920
DMD_Height = 1080
Camera_Width = 2448   # - ? 
Camera_Height = 2048  # - ?

window_width = 2000
window_height = 1300

TAB1 = "  "
TAB2 = "    "


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
     
     
def Upload_DD1_to_DMD(dmd):
     img = np.array(Image.open('patterns//DD_Pat_11.bmp')) # DD20 for courser pattern!
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)  

def Upload_DD2_to_DMD(dmd):
     img = np.array(Image.open('patterns//DD_Pat_22.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)  
     
     
def Upload_DD3_to_DMD(dmd):
     img = np.array(Image.open('patterns//DD_Pat_33.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd) 
     
     
     
     
     
# TI Patterns!!!!!     
def Upload_TI1_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI1.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)
     
def Upload_TI2_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI2.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)  

def Upload_TI3_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI3.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)
     
def Upload_TI4_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI4.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)  

def Upload_TI5_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI5.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)
     
def Upload_TI6_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI6.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)  

def Upload_TI7_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI7.bmp'))
     img1 = img/247
     img11 = img1.astype(np.uint8)
     Upload_Arr_to_DMD(img11, dmd)
     
def Upload_TI8_to_DMD(dmd):
     img = np.array(Image.open('patterns//TI8.bmp'))
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


###################################################### TriggerScope-DAQ Helpers



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


########### TI STUFF!!!
def test_Circle(ix, iy, fx, fy):

    # Use this to find pixel mismatch with the FB pattern uploaded prior to aquisition 
    # This is only used for calibration with the test circle, it will return offset values that will later be used for 
    # converting rectangles into DMD masks 

    dmd_ix = int(ix/2.8) 
    dmd_iy = int(iy/2.8)
    dmd_fx = int(fx/2.7) 
    dmd_fy = int(fy/2.7)
    
    # Real Dimension for circle on DMD 
    
    top = 203
    bottem = 877
    left = 623
    right = 1297
    
    #Compute DIfference: 
    diff_top = top - dmd_iy
    diff_left = left - dmd_ix
    
    print(diff_left, diff_top)
    
    # diff_top gives us the offset we need to add to iy and diff_left for ix
    return diff_left, diff_top 


# Function to convert IMX picture to DMD mask

def convert_pic_to_DMD(ix, iy, fx, fy, diff_left, diff_top):
    global DMD_Height, DMD_Width
    
    #AR_Mismatch_x = 285
    
    dmd_ix = int(ix/2.8) + diff_left 
    dmd_iy = int(iy/2.8) + diff_top 
    dmd_fx = int(fx/2.7) + diff_left
    dmd_fy = int(fy/2.7) + diff_top
    
    all_on_arr = np.ones((DMD_Height, DMD_Width),np.uint8)
    arr = all_on_arr

    for i in range(1920):
        if i < dmd_ix:
            arr[:,i] = 0
        elif i > dmd_fx:
            arr[:,i] = 0

    for j in range(1080):
        if j < dmd_iy:
            arr[j,:] = 0
        elif j > dmd_fy:
            arr[j,:] = 0

   
    plt.imshow(arr)
    plt.show()
    
    # Flip cuz the DMD is actually upside down
    arr2 = np.flipud(arr)

    return arr2

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
            img2 = cv2.imread("Try.jpg")
            cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(255,255,255),thickness=10)
            display_img = img2

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img2 = cv2.imread("Try.jpg")
        cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(255,255,255),thickness=10)
        display_img = img2
        fx = x
        fy = y  
    
    
def calibrate_TI(device, dmd):
    global ix, iy, drawing, display_img, fx, fy, window_width, window_height
    
    # Use this code with the full circle (FB) pattern on the DMD 
    # Try to get the calibration square as close to the circle edges as possible 
    # This must be run before the other TI scripts 

    # Connect to Cam 
    #device = create_device_from_serial_number("220600074")
    
    
    
    # TODO: Load FB Pattern directly from Python instead of using DLP GUI
    # Connect to DMD
    #dmd = dlp6500.dlp6500win(debug=False)
    Upload_FB_to_DMD(dmd)
    
    """
    #Setup Stream
    #tl_stream_nodemap = device.tl_stream_nodemap
    #tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    #tl_stream_nodemap['StreamPacketResendEnable'].value = True
    #tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    new_pixel_format = 'Mono8'
    nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
    nodes['PixelFormat'].value = new_pixel_format
    
    """
    # Reactangle Stuff for TI
    key = -1
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", draw_reactangle_with_drag)

    # Starting Stream and Grabbing IMage 
    device.start_stream()
    image_buffer = device.get_buffer()


    # Long Line to convert image to proper data format for DMD
    nparray = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))

    # Live Box Drawing Stuff for choosing Cells 
    cv2.imwrite("Try.jpg", nparray)
    display_img = cv2.imread("Try.jpg")
    cv2.resizeWindow("Image", window_width, window_height)

    # This runs until ESCAPE Key Pressed 
    while True:
        cv2.imshow("Image", display_img)
        if cv2.waitKey(10) == 27:
            break
    # DMD mask will be displayed, Must Click X button to Continue! 

    # Close out of stream (no need to record while processing)
    device.stop_stream()  
    cv2.destroyAllWindows()

    

    
    # Stop DMD
    dmd.start_stop_sequence("stop")
 


    print("The box drawn has corners (cam dimensions) at: ",ix,iy,fx,fy)
    diff_left, diff_top = test_Circle(ix,iy,fx,fy)
    print("diff left: ", diff_left)
    print("diff Top: ", diff_top)
    
    return diff_left, diff_top



#def match_Offsets(offset_X, offset_Y, R_W, R_H, B_W, B_H


def User_Draws_TI(d_l,d_t, device, dmd):

    global ix, iy, drawing, display_img, fx, fy
    
    # Connect to Cam 
    #device = create_device_from_serial_number("220600074")

    #Setup Stream
    tl_stream_nodemap = device.tl_stream_nodemap
    nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])

    # Reactangle Stuff for TI
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", draw_reactangle_with_drag)
    
    #dmd = dlp6500.dlp6500win(debug=False)
    #Upload_FB_to_DMD(dmd)
    dmd.start_stop_sequence("start")

    # Starting Stream and Grabbing IMage 
    device.start_stream()
    
    #tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    image_buffer = device.get_buffer()
    device.requeue_buffer(image_buffer)
    image_buffer = device.get_buffer()
    
    dmd.start_stop_sequence("stop")
    
    # Long Line to convert image to proper data format for DMD
    nparray = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))


    # Live Box Drawing Stuff for choosing Cells 
    cv2.imwrite("Try.jpg", nparray)
    display_img = cv2.imread("Try.jpg")
    cv2.resizeWindow("Image", window_width, window_height)

    # THis runs until ESCAPE Key Pressed 
    while True:
        cv2.imshow("Image", display_img)
        if cv2.waitKey(10) == 27:
            break
    # DMD mask will be displayed, Must Click X button to Continue! 

    # Close out of stream (no need to record while processing)
    device.stop_stream()  
    cv2.destroyAllWindows()
    #system.destroy_device()


    print("The box drawn has corners at: ",ix,iy,fx,fy)

    diff_left = d_l
    diff_top = d_t


    # Convert Box Drawn with Values from calibratration step 
    arr = convert_pic_to_DMD(ix,iy,fx,fy, diff_left,diff_top)



    # Upload Pattern -> The function in this code has no triggers, it will just display the pattern requested 
    Upload_Arr_to_DMD(arr, dmd)



























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
#device = create_device_from_serial_number("220600074") #Blue
#device = create_device_from_serial_number("224500525") #Red


#dmd1 = dlp6500.dlp6500win(debug=False)  
#dmd2 = dlp6500.dlp6500win2(debug=False) 

# Connect to uManager/TriggerScope
#core = Core()

# Connect to Arduino for Electrical Recording..
#arduino = serial.Serial('COM8', 115200, timeout=.1)

Upload_TI4_to_DMD(dmd1)



#Upload_FB_to_DMD(dmd1)
#Upload_FB_to_DMD(dmd2)



#Upload_DD1_to_DMD(dmd1)
#Upload_DD2_to_DMD(dmd1)
#Upload_DD3_to_DMD(dmd1)

def run_SIM(device, dmd):
    pixel_format = PixelFormat.Mono8
    # Get MMJ
    bridge = Bridge()
    mmc = bridge.get_core()

    
    iter = 0
    while iter < 5:
        # Pat 1
        i = 1
        print("pat1")
        Upload_DD1_to_DMD(dmd)
        time.sleep(3)
        device.start_stream(1)
        buffer = device.get_buffer()
        converted = BufferFactory.convert(buffer, pixel_format)
        writer = Writer()
        writer.pattern = 'SIM_images/image_<count>' + str(iter) + str(i) + '.jpg'
        writer.save(converted)
        BufferFactory.destroy(converted)
        device.requeue_buffer(buffer)
        device.stop_stream()
        
        
        # Pat 2
        i = 2
        print("pat2")
        Upload_DD2_to_DMD(dmd)
        time.sleep(3)
        device.start_stream(1)
        buffer = device.get_buffer()
        converted = BufferFactory.convert(buffer, pixel_format)
        writer = Writer()
        writer.pattern = 'SIM_images/image_<count>' + str(iter) + str(i) + '.jpg'
        writer.save(converted)
        BufferFactory.destroy(converted)
        device.requeue_buffer(buffer)    
        device.stop_stream()
 
        # Pat 3
        i = 3
        print("pat3")
        Upload_DD3_to_DMD(dmd)
        time.sleep(3)
        device.start_stream(1)
        buffer = device.get_buffer()
        converted = BufferFactory.convert(buffer, pixel_format)
        writer = Writer()
        writer.pattern = 'SIM_images/image_<count>' + str(iter) + str(i) + '.jpg'
        writer.save(converted)
        BufferFactory.destroy(converted)
        device.requeue_buffer(buffer)   
        device.stop_stream()
        
        
        
        iter = iter + 1

    print("DONE")
    #Clean Up
    device.stop_stream()

    # Destroy Device
    system.destroy_device(device)   
    dmd.start_stop_sequence("stop") 


#run_SIM(device,dmd1)

#diff_left, diff_top = calibrate_TI(device, dmd2)
#print("Calibration Complete")
#User_Draws_TI(diff_left,diff_top, device, dmd2)





#Upload_FB_to_DMD(dmd1)
#Upload_FB_to_DMD(dmd2)












