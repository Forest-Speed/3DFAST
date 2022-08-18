# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 18:12:34 2022

@author: Forest Speed
"""

# General Stuff
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import os
from skimage import io
from PIL import Image
from matplotlib import pyplot as plt


import time
import ctypes
import numpy as np

from arena_api.system import system
from arena_api.buffer import *
from arena_api.__future__.save import Writer
from arena_api.buffer import BufferFactory
from arena_api import enums
from arena_api.enums import PixelFormat
#from arena_api.__future__.save import Recorder
from arena_api.callback import callback, callback_function

from pycromanager import Bridge

import PySimpleGUI as sg

# DMD Stuff -> DLP GUI Must be Closed! 
from mcsim.expt_ctrl import dlp6500


# For Multi-Threading 
import queue
import threading
from multiprocessing import Value

print("IMPORTS DONE")

#############################################################################
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








# Connect to Camera 
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


def set_ROI(device, width, height, offset_X, offset_Y):
    
    nodemap = device.nodemap
    #nodes = nodemap.get_node(['Width', 'Height','OffsetX','OffsetY',])
    

    width_node = nodemap['Width']

    # get a value that aligned with node increments
    while width % width_node.inc:
        width -= 1
    nodemap['Width'].value = width

    height_node = nodemap['Height']

    # get a value that aligned with node increments
    while height % height_node.inc:
        height -= 1
    nodemap['Height'].value = height
    
        
    offset_X_node = nodemap['OffsetX']

    # get a value that aligned with node increments
    while offset_X % offset_X_node.inc:
        offset_X -= 1
    nodemap['OffsetX'].value = offset_X

    offset_Y_node = nodemap['OffsetY']

    # get a value that aligned with node increments
    while offset_Y % offset_Y_node.inc:
        offset_Y -= 1
    nodemap['OffsetY'].value = offset_Y

    
def reset_ROI(device):
    
    nodemap = device.nodemap
    nodemap['OffsetX'].value = 0
    nodemap['OffsetY'].value = 0
    nodemap['Width'].value = nodemap['Width'].max
    nodemap['Height'].value = nodemap['Height'].max
    
    
def set_Cam_Trigs(device, exposure_Time):
    

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


def rapid_Record(device, num_Frames):
    

    device.nodemap['AcquisitionMode'].value = 'Continuous'
    
    nodemap = device.nodemap
    nodemap['PixelFormat'].value = PixelFormat.Mono8
    
    # automate the calculation of max FPS whenever the device settings change
    nodemap['AcquisitionFrameRateEnable'].value = True

    """
    set FPS node to max FPS which was set to be automatically calculated
    base on current device settings
    """
    nodemap['AcquisitionFrameRate'].value = nodemap['AcquisitionFrameRate'].max

    # max FPS according to the current settings
    nodemap['DeviceStreamChannelPacketSize'].value = nodemap['DeviceStreamChannelPacketSize'].max

    # Calibration Function with Circle Pattern (FB) for TI Codes 
    
    total_images = num_Frames

    with device.start_stream(1):
        print('Stream started')
        
        recorder = Recorder(nodemap['Width'].value,
                        nodemap['Height'].value,
                        nodemap['AcquisitionFrameRate'].value,
                        threaded=True)
    
    
        recorder.codec = ('h264', 'avi', 'mono8')  # order does not matter
        recorder.pattern = 'My_vid<count>.avi'
        recorder.open()
        print('recorder opened')



        for itr_count in range(total_images):

            buffer = device.get_buffer()


            recorder.append(buffer)
            print(f'Image buffer {itr_count} appended to video')


            device.requeue_buffer(buffer)


    recorder.close() 
    print('recorder closed')

    # video path
    print(f'video saved {recorder.saved_videos[-1]}')

    # approximate length of the video
    video_length_in_secs = total_images / \
        nodemap['AcquisitionFrameRate'].value

    print(f'video length ~= {video_length_in_secs: .3} seconds')
    print(nodemap['Width'].value,
                    nodemap['Height'].value,
                    nodemap['AcquisitionFrameRate'].value)
    
    
    
#device = create_device_from_serial_number("220600074")    
#rapid_Record(device, 1000)
    

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



# Function to Upload Array to DMD

def Upload_Arr_to_DMD(arr,dmd):
    pat = arr
    exposure_t = 150
    dark_t = 0
    triggered = False
    img_inds, bit_inds = dmd.upload_pattern_sequence(pat, exposure_t, dark_t, triggered, clear_pattern_after_trigger = False, bit_depth = 1, num_repeats = 0)
    dmd.set_pattern_sequence(img_inds, bit_inds, exposure_t, dark_t, triggered=False, clear_pattern_after_trigger= False, bit_depth=1, num_repeats=0, mode='on-the-fly')
    dmd.start_stop_sequence("start")

def Upload_FB_to_DMD(dmd):
    img = np.array(Image.open('patterns//FB.bmp'))
    img1 = img/247
    img11 = img1.astype(np.uint8)
    Upload_Arr_to_DMD(img11, dmd)
    
def run_Z_Stack(start, stop, step_size, dmd, device):
    
    # Upload All On Pat
    Upload_FB_to_DMD(dmd)
    
    # Get MMJ
    bridge = Bridge()
    mmc = bridge.get_core()
   

    z = start
    z_Step = step_size
    
    pixel_format = PixelFormat.Mono8
    device.start_stream()
    print("Stream Started")
    
    iter = 0
    while z < stop:
        mmc.set_property("TS_DAC03", "Volts", z) 
        buffer = device.get_buffer()

        converted = BufferFactory.convert(buffer, pixel_format)
    
        #Prepare Image Writer
        writer = Writer()
       
        writer.pattern = 'images/image_<count>' + str(iter) + '.jpg'
        

        print(iter,'_:_',z)
        
        
        
        #Save Converted Buffer
        writer.save(converted)
        
    
        #Destroy Converter Buffer
        BufferFactory.destroy(converted)
        device.requeue_buffer(buffer)
        iter = iter + 1
        z = z + z_Step
        
    print("DONE")
    #Clean Up
    device.stop_stream()

    # Destroy Device
    #system.destroy_device(device)   
    dmd.start_stop_sequence("stop") 
    
@callback_function.device.on_buffer
def print_buffer(buffer, *args, **kwargs):
    
    with threading.Lock():
        print(f'{TAB2}{TAB1}Buffer callback triggered'
              f'(frame id is {buffer.frame_id})')



# MultiThreading Functions 
def get_multiple_image_buffers(device, buffer_queue, is_more_buffers):

    number_of_buffers = 3

    device.start_stream(number_of_buffers)
    print(f'Stream started with {number_of_buffers} buffers')

    print(f'\tGet {number_of_buffers} buffers in a list')
    buffers = device.get_buffer(number_of_buffers)
   
    # Print image buffer info
    for count, buffer in enumerate(buffers):
        """
        print(f'\t\tbuffer{count:{2}} received | '
              f'Width = {buffer.width} pxl, '
              f'Height = {buffer.height} pxl, '
              f'Pixel Format = {buffer.pixel_format.name}')
        """
        buffer_queue.put(BufferFactory.copy(buffer))
        time.sleep(0.1)


    device.requeue_buffer(buffers)
    print(f'Requeued {number_of_buffers} buffers')

    device.stop_stream()
    print('Stream stopped')
    is_more_buffers.value = 0


def save_image_buffers(buffer_queue, is_more_buffers):

    # SAVE AS TiF


    writer = Writer()
    count = 1
    t = str(time.time())
    while is_more_buffers.value or not buffer_queue.empty():
        while(not buffer_queue.empty()):
            buffer = buffer_queue.get()
            writer.save(buffer, pattern=f"Images/image_{t}_{count}.jpg")
            print(f"Saved image {count}")
            count = count + 1
        print("Queue empty, waiting 1s")
        time.sleep(1)


def set_Cam():
    
    # Get Camera 
    device = create_device_from_serial_number("220600074")
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
    
    return device, new_pixel_format, exposure_time_initial


def lock_Exposure(device):
    nodemap = device.nodemap
    nodes = nodemap.get_node(['ExposureAuto', 'ExposureTime'])

    nodes['ExposureAuto'].value = "Off"
    
    
    
    


def set_DMD_SIM(exposure_time_cam):
    
    # import tif file into numpy array & Convert to Binary
    pats = io.imread('patterns//OS_FS_EXTRA_EXTENDED.tif')                                         #!!!!!!!! EXTRA for Z slices! 
    pats[np.where(pats!=0)] = 1
    # Get DMD & Upload Pats
    dmd = dlp6500.dlp6500win(debug=False)
    print(dmd)
    
    exposure_t = int(exposure_time_cam - 250)
    dark_t = 0
    triggered = True
    
    img_inds, bit_inds = dmd.upload_pattern_sequence(pats, exposure_t, 0, triggered, clear_pattern_after_trigger = True, bit_depth = 1, num_repeats = 0)
    print("Upload Complete")
    
    
    dmd.set_pattern_sequence(img_inds, bit_inds, exposure_t, dark_t, triggered, clear_pattern_after_trigger= True, bit_depth=1, num_repeats=0, mode='on-the-fly')
    dmd.start_stop_sequence("start")
    
    print("DMD IS INITIALIZED")
    return dmd


def set_MM_DMD_Trigs():
    
    # Connect uManager
    bridge = Bridge()
    mmc = bridge.get_core()
    
    #Trig1
    mmc.set_property("TS_DAC01", "Blanking","On") 
    mmc.set_property("TS_DAC01", "State","1")
    mmc.set_property("TS_DAC01", "Volts", str(3.3))

    #Trig2
    #mmc.set_property("TS_DAC02", "Blanking","Off") 
    mmc.set_property("TS_DAC02", "State","1")
    mmc.set_property("TS_DAC02", "Volts", str(3.3))


def turn_Off__MM_Trigs():
    # Connect uManager
    bridge = Bridge()
    mmc = bridge.get_core()
    

    mmc.set_property("TS_DAC01", "State","0")

    mmc.set_property("TS_DAC02", "State","0")



def set_Mode_Multi(device):
    nodemap = device.nodemap
    nodes = nodemap.get_node(['AcquisitionMode','AcquisitionFrameCount'])
    
    
    nodes['AcquisitionMode'].value = "MultiFrame"
    nodes['AcquisitionFrameCount'].value = 3


def set_Mode_Cont(device):
    nodemap = device.nodemap
    nodes = nodemap.get_node(['AcquisitionMode'])
    nodes['AcquisitionMode'].value = "Continuous"


# SIM on one plane func:
def Run_SIM_MultiThread(device, dmd, z_Volts):
    bridge = Bridge()
    mmc = bridge.get_core()
    mmc.set_property("TS_DAC03", "Volts", z_Volts)
    
    set_Mode_Multi(device)
    
    # Multithreading Variables
    is_more_buffers = Value('i', 1)
    buffer_queue = queue.Queue()

    #handle = callback.register(device, print_buffer)
       

    # Run Multithreading w/ Hardware Controlling SIM Pats
    acquisition_thread = threading.Thread(target=get_multiple_image_buffers, args=(device, buffer_queue, is_more_buffers))
    acquisition_thread.start()
    time.sleep(1)
    save_image_buffers(buffer_queue, is_more_buffers)
    acquisition_thread.join()

    #callback.deregister(handle)
    
    # Disconnect CAM
    #system.destroy_device()
    
    # Disconnect DMD
    #dmd.start_stop_sequence("stop")
    
      
# Rectangle Drawing Function used for TI 


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
    #Upload_FB_to_DMD(dmd)
    
    """
    #Setup Stream
    tl_stream_nodemap = device.tl_stream_nodemap
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True
    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
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



def User_Draws_TI(d_l,d_t, device, dmd):
    
    # TODO: Add Multithreading to TI , make calibratino better
    
    
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


    # Next -> Reconnect -> Switch ROI and Aquire! 

    # Connect to Cam 
    #device = create_device_from_serial_number("220600074")

    # ROI's 

    width = fx - ix 
    height = fy - iy
    offset_X = ix
    offset_Y = iy
    

    pixel_Format = 'Mono8'
    
    set_ROI(device, width, height, offset_X, offset_Y)


    dmd.start_stop_sequence("start")
    
    """
    # Multithread in recorder -> Returns MP4 at MAX FPS 
    num_Frames = 1000
    
    rapid_Record(device, num_Frames)
    
    
    
    
    
    
    
    
    
    """
    
    # MULTITHREADING IN WRITER !

    # TODO -> Multithreading for Speed !
    # TODO -> Add LED Piezo Control Stuff !
    # TODO -> Buffer order is BAD! Need to figure this out for SIM !!!!!!
    pixel_Format = 'Mono8'
    #device.start_stream(1)
    
    # Multithreading Variables
    is_more_buffers = Value('i', 1)
    buffer_queue = queue.Queue()

        
    # Run Multithreading w/ Hardware Controlling SIM Pats
    acquisition_thread = threading.Thread(target=get_multiple_image_buffers, args=(device, buffer_queue, is_more_buffers))
    acquisition_thread.start()
    time.sleep(1)
    save_image_buffers(buffer_queue, is_more_buffers)
    acquisition_thread.join()
    """
    
    
    

    
    # No Multithread! 
    
    iter = 0 
    while  iter < 1:
        # Grab images --------------------------------------------------------
        buffer = device.get_buffer()
        converted = BufferFactory.convert(buffer, pixel_Format)

    
        #Prepare Image Writer

        writer = Writer()
        writer.pattern = 'images/image_<count>' + str(iter) + '.jpg'
    
        #Save Converted Buffer
        writer.save(converted)
        #writer.save(buffer_Mono16)
        #Destroy Converter Buffer
        BufferFactory.destroy(converted)
 
        device.requeue_buffer(buffer)
        iter = iter + 1
    
        print("Took Photo")
    """
    device.stop_stream()
    dmd.start_stop_sequence("stop")
    #system.destroy_device() 



def get_image_buffers(device, mmc, is_color_camera=False):
    
    global dmd, flag, start_Pos, stop_Pos, current_Pos
    current_Pos = 0
    start_Pos = 0
    stop_Pos = 0
    dmd = dlp6500.dlp6500win(debug=False) 

    key = -1
    flag = 0
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    nodemap = device.nodemap
    set_Mode_Cont(device)
    device.start_stream()

    # Z Slider callback function
    def z_Slide(x):
        
        # Z Position Stuff
        pos_mV = cv2.getTrackbarPos("Z Position", "Image")
        pos_V = pos_mV/100
        mmc.set_property("TS_DAC03", "Volts", str(pos_V) )
    
    # arguments: trackbar_name, window_name, default_value, max_value, callback_fn
    cv2.createTrackbar("Z Position", "Image", 0, 1000, z_Slide)
    
    
    # LED Slider callback function
    def LED_Slide(x):
        
        pos_LED = cv2.getTrackbarPos("LED Strength", "Image")
        pos_V_LED = pos_LED/100
        #print(pos_V_LED)
        mmc.set_property("TS_DAC08", "Volts", str(pos_V_LED) )
    
    # arguments: trackbar_name, window_name, default_value, max_value, callback_fn
    cv2.createTrackbar("LED Strength", "Image", 0, 330, LED_Slide)
    
    """ 

    def gain_Slide(x):
        pos_Gain = cv2.getTrackbarPos("Gain Slide", "Image")
        pos_G = float(pos_Gain)
        nodemap['Gain'].value = pos_G
       
    cv2.createTrackbar("Gain Slide", "Image", 0, 48, gain_Slide)
    """
    
    def gain_Slide(x):
        pos_Gain = cv2.getTrackbarPos("Gain Slide", "Image")
        pos_G = float(pos_Gain)
        nodemap['Gain'].value = pos_G
       
    cv2.createTrackbar("Gain Slide", "Image", 0, 48, gain_Slide)
    
 
    def start_Button_Press(*args):
        global start_Pos
        start_Pos = cv2.getTrackbarPos("Z Position", "Image")/100
        print("Start Position: ", start_Pos)
    
    cv2.createButton("Register Start Z", start_Button_Press, None, cv2.QT_PUSH_BUTTON,1)
   
    
    def end_Button_Press(*args):
        global stop_Pos
        stop_Pos = cv2.getTrackbarPos("Z Position", "Image")/100
        print("End Position: ", stop_Pos)
    
    cv2.createButton("Register Stop Z", end_Button_Press, None, cv2.QT_PUSH_BUTTON,1)
    
    def display_All_On_DMD(*args):
        
        global dmd
        print(dmd)
        Upload_FB_to_DMD(dmd)
        
    cv2.createButton("All on DMD", display_All_On_DMD, None, cv2.QT_PUSH_BUTTON,1)
    
    def stop_DMD(*args):
        
        global dmd
        dmd.start_stop_sequence("stop")
        
        
    cv2.createButton("Stop DMD", stop_DMD, None, cv2.QT_PUSH_BUTTON,1)
    
    def run_Z_Stack(*args):
        global start_Pos, stop_Pos, dmd, flag
        flag = 1
        key == ord("q")
        
    cv2.createButton("Run Z Stack", run_Z_Stack, None, cv2.QT_PUSH_BUTTON,1)
    
    
    def run_SIM_Frame(*args):
       global current_Pos, flag
       set_MM_DMD_Trigs()
       
       current_Pos = cv2.getTrackbarPos("Z Position", "Image")/100
       
       flag = 2
       key == ord("q")
       
    cv2.createButton("Run SIM at this Z", run_SIM_Frame, None, cv2.QT_PUSH_BUTTON,1)
   
    def calibrate_Targets(*args):
        global flag
        #cv2.setMouseCallback("Image", draw_reactangle_with_drag)
        flag = 3
        key == ord("q")
    
    cv2.createButton("Calibrate TI", calibrate_Targets, None, cv2.QT_PUSH_BUTTON,1)
    
    def z_SIM(*args):
        global flag, current_Pos, start_Pos, stop_Pos
        
        #set_MM_DMD_Trigs()
    
        current_Pos = cv2.getTrackbarPos("Z Position", "Image")/100
    
        flag = 4
        key == ord("q")
        
    cv2.createButton("Z SIM", z_SIM, None, cv2.QT_PUSH_BUTTON,1)

    
    while True:
        image_buffer = device.get_buffer()  # optional args
        nparray = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))


        display_img = cv2.cvtColor(nparray, cv2.COLOR_GRAY2BGR)
        
        cv2.resizeWindow("Image", window_width, window_height)
        cv2.imshow("Image", display_img)
                
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        #BufferFactory.destroy(converted_image)
        device.requeue_buffer(image_buffer)

    cv2.destroyAllWindows()
    
    device.stop_stream()
    if flag == 0:
        print("No Aquisition Selected")
        #system.destroy_device()
        
    return flag, start_Pos, stop_Pos, current_Pos, dmd,  







#dmd = set_DMD_SIM(1643000.56)





device = create_device_from_serial_number("220600074")

bridge = Bridge()
mmc = bridge.get_core()



flag, start_Pos, stop_Pos, current_Pos, dmd  = get_image_buffers(device, mmc,  is_color_camera=False)

# Z Stack
if flag == 1:
    
    step_size = .01 #!!!!!!!!!!!!
    
    
    
    
    run_Z_Stack(start_Pos, stop_Pos, step_size, dmd, device)

# SIM
if flag == 2:
    device, pixel_format, exposure_time_initial = set_Cam()
    set_Cam_Trigs(device)
    z_Volts = current_Pos
    set_MM_DMD_Trigs()
    dmd = set_DMD_SIM(exposure_time_initial)
    Run_SIM_MultiThread(device, dmd, z_Volts)
    dmd.start_stop_sequence("stop")
    turn_Off__MM_Trigs()

if flag == 3:
    diff_left, diff_top = calibrate_TI(device, dmd)
    print("Calibration Complete")
    User_Draws_TI(diff_left,diff_top, device, dmd)

if flag == 4:
    device, pixel_format, exposure_time_initial = set_Cam()
    
    device.stop_stream()
    #set_Mode_Multi(device)
    
    lock_Exposure(device)
    set_Cam_Trigs(device, exposure_time_initial)
    
    z_Volts = start_Pos
    
    set_MM_DMD_Trigs()
    
    """
    mmc.set_property("TS_DAC01", "State","1")
    mmc.set_property("TS_DAC01", "Volts", str(3.3))
    mmc.set_property("TS_DAC02", "State","1")
    mmc.set_property("TS_DAC02", "Volts", str(3.3)) 
    """
    
    dmd = set_DMD_SIM(exposure_time_initial)

    
    
    
    
    while z_Volts < stop_Pos:
        
        Run_SIM_MultiThread(device, dmd, z_Volts)
        

        
        #restart pattern
       
        time.sleep(.1)

    
        
        
        
        #Move to next z
        z_Volts = z_Volts + .02 # STEP SIZE = .01 !!!!
    
    
    turn_Off__MM_Trigs()
    dmd.start_stop_sequence("stop")





reset_ROI(device)
system.destroy_device() 

# 45 um poer 1V
# 4.5um is .1
# .05V is 2.25 um 
# Do LED measurements 












































































































