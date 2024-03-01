import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import axes
#from skimage.feature import graycomatrix, graycoprops

import skimage as ski
import cv2 as cv
#import pickle as pkl
import xarray as xr
###############################

#Load data

metrics = xr.open_dataset('/home/nmitchell/GLCM/testing_data.nc')

# Input number of convection files and tile size
#Possible Examples: 16, 21, 25, 50, 55, 61
#NEW: 10, *11*, 19
num  = len(metrics.coords['Sample'])
num1 = 0
num2 = 1

tile_size = 4
num_rows  = int(256/tile_size)
num_cols  = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["red", "black"])
grn_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["mediumseagreen", "black"])

# IMAGES ORDER:
# Original Image # Convolved Im  # Ground Truth     # Infrared Image #                 # Convolved Image    #                     # Expanded Ground Truth
# GLCM - OG Im   # Min Mask      # Min App OG GLCM  # Infrared Mask  # IR Mask Applied # GLCM - Conv. Image # Min App Conv. GLCM  #
#                # Mean Mask     # Mean App OG GLCM #                # IR Mask Applied #                    # Mean App Conv. GLCM #

for n in range(num1, num2):
    isamp = n

    #Import & Reshape Original Image, Convolved Image, Infrared Image, and Ground Truth
    og_image  = np.array(metrics.Original_Image.sel(Sample  = isamp))
    og_length = int(np.sqrt(len(og_image)))
    og_image  = og_image.reshape(og_length, og_length)

    cv_image  = np.array(metrics.Convolved_Image.sel(Sample = isamp))
    cv_length = int(np.sqrt(len(cv_image)))
    cv_image  = cv_image.reshape(cv_length, cv_length)

    ir_image  = np.array(metrics.Infrared_Image.sel(Sample  = isamp))
    ir_length = int(np.sqrt(len(ir_image)))
    ir_image  = ir_image.reshape(ir_length, ir_length)

    gt_image  = np.array(metrics.Ground_Truth.sel(Sample = isamp))
    gt_length = int(np.sqrt((len(gt_image)/4)))
    gt_image  = gt_image.reshape(gt_length, gt_length, 4)

    #Import & Reshape GLCM-Related Metrics
    og_GLCM     = np.array(metrics.Original_GLCM.sel(Sample = isamp))
    glcm_length = int(np.sqrt(len(og_GLCM)))

    og_GLCM     = og_GLCM.reshape(glcm_length, glcm_length)
    conv_GLCM   = np.array(metrics.Convolved_GLCM.sel(Sample = isamp)).reshape(glcm_length, glcm_length)

    #Import & Reshape Mask-Related Metrics
    min_mask    = np.array(metrics.Min_Mask.sel(Sample = isamp))
    mask_length = int(np.sqrt(len(min_mask)))

    min_mask    = min_mask.reshape(mask_length, mask_length)
    mean_mask   = np.array(metrics.Mean_Mask.sel(Sample = isamp)).reshape(mask_length, mask_length)
    ir_mask     = np.array(metrics.Infrared_Image_Mask.sel(Sample = isamp)).reshape(mask_length, mask_length)

    #Import & Reshape Applied-Mask-Related Metrics
    min_mask_app  = np.array(metrics.Min_Mask_Applied.sel(Sample = isamp)).reshape(mask_length, mask_length)
    mean_mask_app = np.array(metrics.Mean_Mask_Applied.sel(Sample = isamp)).reshape(mask_length, mask_length)

    ir_mask_min   = np.array(metrics.Infrared_Mask_Applied_to_Min_Mask.sel(Sample = isamp)).reshape(mask_length, mask_length)
    ir_mask_mean  = np.array(metrics.Infrared_Mask_Applied_to_Mean_Mask.sel(Sample = isamp)).reshape(mask_length, mask_length)

    min_app_c     = np.array(metrics.Min_Mask_Applied_Convolved_Image.sel(Sample = isamp)).reshape(mask_length, mask_length)
    mean_app_c    = np.array(metrics.Min_Mask_Applied_Convolved_Image.sel(Sample = isamp)).reshape(mask_length, mask_length)

    ones = np.ones((int(256/tile_size), int(256/tile_size)))

    fig, ax = plt.subplots(3,7)

    ### COLUMN 1 ###
    ax[0,0].set_title("Original Image")
    ax[0,0].imshow(og_image, cmap = 'gray', origin = 'lower')

    ax[1,0].set_title("Original GLCM")
    ax[1,0].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[1,0].imshow(ones, alpha = og_GLCM, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))
    #ax[1,0].set_xlabel("Max: " + "%.2f" % max(np.array(metrics["Max/Min Contrast"])[:,0]) + "\n Min: " + "%.2f" %  min(np.array(metrics["Max/Min Contrast"])[:,1]))

    ax[2,0].set_visible(False)

    ### COLUMN 2 ###
    ax[0,1].set_title("Convolved Image")
    ax[0,1].imshow(cv_image, cmap = 'gray', origin = 'lower')

    ax[1,1].set_title("Min Mask")
    ax[1,1].imshow(og_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
    ax[1,1].imshow(min_mask, cmap = grn_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,1].set_title("Mean Mask")
    ax[2,1].imshow(og_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
    ax[2,1].imshow(mean_mask, cmap = grn_cm, origin = 'lower', extent = (0,256,0,256))

    ### COLUMN  3 ###
    ax[0,2].set_title("Supposed Ground\nTruth")
    ax[0,2].imshow(gt_image, cmap = 'gray', origin = 'lower')

    ax[1,2].set_title("Min Mask Applied")
    ax[1,2].imshow(og_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
    ax[1,2].imshow(ones, alpha = min_mask_app, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,2].set_title("Mean Mask Applied")
    ax[2,2].imshow(og_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
    ax[2,2].imshow(ones, alpha = mean_mask_app, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ### COLUMN 4 ###
    ax[0,3].set_title("Infrared Image")
    ax[0,3].imshow(ir_image, cmap = "gray", origin = "lower")

    ax[1,3].set_title("Infrared Mask")
    ax[1,3].imshow(ir_image, cmap = "gray", origin = "lower")
    ax[1,3].imshow(ir_mask, cmap = grn_cm, origin = 'lower') #, extent = (0,256,0,256))
    ax[1,3].set_xlabel("Threshold Value: 250K")

    ax[2,3].set_visible(False)

    ### COLUMN 5 ###
    ax[0,4].set_visible(False)

    ax[1,4].set_title("IR Mask Applied")
    ax[1,4].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[1,4].imshow(ones, alpha = ir_mask_min, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,4].set_title("IR Mask Applied")
    ax[2,4].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[2,4].imshow(ones, alpha = ir_mask_mean, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ### COLUMN 6 ###
    ax[0,5].set_title("Convolved Image")
    ax[0,5].imshow(cv_image, cmap = 'gray', origin = 'lower')

    ax[1,5].set_title("Convolve GLCM")
    ax[1,5].imshow(og_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
    #ax[1,5].set_xlabel("Max: " + "%.2f" % max(np.array(metrics["Max/Min Contrast C"])[:,0]) + "\n Min: " + "%.2f" % min(np.array(metrics["Max/Min Contrast C"])[:,1]))
    ax[1,5].imshow(ones, alpha = conv_GLCM, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,5].set_visible(False)

    ### COLUMN 7 ###
    ax[0,6].set_visible(False)

    ax[1,6].set_title("Min Mask Applied")
    ax[1,6].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[1,6].imshow(ones, alpha = min_app_c, cmap = red_cm, origin = 'lower',extent = (0,256,0,256))

    ax[2,6].set_title("Mean Mask Applied")
    ax[2,6].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[2,6].imshow(ones, alpha = mean_app_c, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    for i in range (0,3):
        for j in range (0,7):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    #fig = plt.gcf()
    #fig.set_size_inches((8.5, 11), forward=False)
    #plt.tight_layout()
    plt.show()
#    filepath = r'/home/nmitchell/GLCM/Images-Contrast/'
#    filepath += 'Contrast_' + str(isamp) + ".png"
#    fig.savefig(filepath)
#    print(filepath)
#    fig.close()
