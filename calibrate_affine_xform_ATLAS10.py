"""
Example showing how to determine the affine calibration between the DMD and camera space
"""
import datetime
import numpy as np
from pathlib import Path
import json
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mcsim.analysis import fit_dmd_affine, mm_io
import mcsim.analysis.dmd_patterns as dmd



from matplotlib.patches import Rectangle
from localize_psf import affine, rois, fit


# atlas10 width = 67.07um
# atlas20 height = 56.12 um
# Pixel pitch - 2.74 um 


# ###########################
# set image data location
# ###########################

# CALIBRATION IMAGE FILE NAME!!!! CHANGE EVERY TIME!!!
img_fname = Path("data", "test_calib_1220.tiff")
channel_labels = ["red"]
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H;%M;%S")
save_dir = img_fname.parent / f"{time_stamp:s}_affine_calibration"


# ###########################
# set guesses for three "spots" from DMD pattern
# ###########################


#centers_init = [[1039, 918], [982, 976], [1091, 979]]
#indices_init = [[10, 16], [9, 16], [10, 15]] #[dmd short axis, dmd long axis]



#centers_init = [[1054,1328,], [880,1328],  [882,1156] ] # [Y,X] BACKWARDS! WORKED WITH 12_16 !
indices_init = [[8, 16], [9, 16], [9,15]] #[dmd short axis, dmd long axis] WORKED WITH 12_16 !

centers_init = [[1056,1454,], [884,1458],  [886,1286] ]




cents = np.array(centers_init)



# ###########################
# set other parameters for fitting
# ###########################

roi_size = 100 #  May need to change this 




options = {'cam_pix': 2.74e-6,
           #'cam_pix': 6.5e-6,
           'dmd_pix': 7.56e-6,
           'dmd2cam_mag_expected': 180 / 18 * 18 / 180,
           #'dmd2cam_mag_expected': 180 / 300 * 400 / 200,
           #'cam_mag': 100
           'cam_mag': 100
             }

def sigma_pix(wl1, wl2, na, cam_mag):
    print("Hello")
    return np.sqrt(wl1**2 + wl2**2) / 2 / na / (2 * np.sqrt(2 * np.log(2))) / (options["cam_pix"] / cam_mag)

# load DMD pattern and dmd_centers
dmd_size = [1920, 1080]
masks, radii, pattern_centers = dmd.get_affine_fit_pattern(dmd_size)
mask = masks[1]


# ###########################
# perform affine calibration for each channel and plot/export results
# ###########################
affine_summary = {}

for nc in range(len(channel_labels)):
    img, _ = mm_io.read_tiff(img_fname, slices=nc)
    #img, _ = mm_io.read_tiff(img_fname)
    ii = img 
    
    img_T = np.transpose(img,(2,0,1))
    
   

    

    #img = img[0]
    img = img_T[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax.plot(cents[:,1],cents[:,0], 'rx')
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(mask)
    dmd_cents_plot = np.array([pattern_centers[i[0],i[1]] for i in indices_init])
    
    ax2.plot(dmd_cents_plot[:,0], dmd_cents_plot[:,1], 'rx')
    plt.show()
    
    
    
    
    #img = np.uint16(img)
    #img = np.flipud(img)
    
    affine_xform_data, figh = fit_dmd_affine.estimate_xform(img,
                                                            mask,
                                                            pattern_centers,
                                                            centers_init,
                                                            indices_init,
                                                            options,
                                                            roi_size=roi_size,
                                                            export_fname=f"affine_xform_{channel_labels[nc]:s}",
                                                            export_dir=save_dir,
                                                            chi_squared_relative_max=3,
                                                            figsize=(20, 12))

    affine_summary[channel_labels[nc]] = affine_xform_data["affine_xform"]

# save summary results
affine_summary["transform_direction"] = "dmd to camera"
affine_summary["processing_time_stamp"] = time_stamp
affine_summary["data_time_stamp"] = ""
affine_summary["file_name"] = str(img_fname)

fname_summary = save_dir / "dmd_affine_transformations.json"
with open(fname_summary, "w") as f:
    json.dump(affine_summary, f, indent="\t")

plt.show()























