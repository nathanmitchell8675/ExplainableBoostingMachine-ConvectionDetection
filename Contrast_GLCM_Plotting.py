import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import axes
from skimage.feature import graycomatrix, graycoprops

import skimage as ski
import cv2 as cv
import pickle as pkl
###############################

#Load data
filepath = r'/home/nmitchell/GLCM/metrics'
with open(filepath, 'rb') as file:
    metrics = pkl.load(file)

# Input number of convection files and tile size
#Possible Examples: 16, 21, 25, 50, 55, 61
#NEW: 10, *11*, 19
num= 204
num1 = 21
num2 = 204
tile_size = 8
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["red", "black"])
grn_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["mediumseagreen", "black"])

for n in range(num1, num2):
    isamp = n

    # og pic # convolve  # ground truth    # convolve       #                       # Infrared Image
    # glcm   # min mask  # min on og glcm  # convolve glcm  # min on convolve glcm  # Infrared Mask
    #        # mean mask # mean on og mask #                # mean on convolve glcm #

    #Import variables that are used more than once
    og_image = metrics["Original Image"][isamp]
    cv_image = metrics["Convolved Image"][isamp]
    ir_image = metrics["Infrared Image"][isamp]

    ones = np.ones((int(256/tile_size), int(256/tile_size)))

    fig, ax = plt.subplots(3,7)

    ### COLUMN 1 ###
    ax[0,0].set_title("Original Image")
    ax[0,0].imshow(og_image, cmap = 'gray', origin = 'lower')

    ax[1,0].set_title("Original GLCM")
    ax[1,0].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[1,0].imshow(ones, alpha = metrics["Contrast Values"][isamp], cmap = red_cm, origin = 'lower', extent = (0,256,0,256))
    #ax[1,0].set_xlabel("Max: " + "%.2f" % max(np.array(metrics["Max/Min Contrast"])[:,0]) + "\n Min: " + "%.2f" %  min(np.array(metrics["Max/Min Contrast"])[:,1]))

    ax[2,0].set_visible(False)

    ### COLUMN 2 ###
    ax[0,1].set_title("Convolved Image")
    ax[0,1].imshow(cv_image, cmap = 'gray', origin = 'lower')

    ax[1,1].set_title("Min Mask")
    ax[1,1].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[1,1].imshow(metrics["Tile Min Mask"][isamp], cmap = grn_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,1].set_title("Mean Mask")
    ax[2,1].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[2,1].imshow(metrics["Tile Mean Mask"][isamp], cmap = grn_cm, origin = 'lower', extent = (0,256,0,256))

    ### COLUMN  3 ###
    ax[0,2].set_title("Supposed Ground\nTruth")
    ax[0,2].imshow(metrics["Ground Truth"][isamp], cmap = 'gray', origin = 'lower')

    ax[1,2].set_title("Min Mask Applied")
    ax[1,2].imshow(og_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
    ax[1,2].imshow(ones, alpha = metrics["Min Mask Applied"][isamp], cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,2].set_title("Mean Mask Applied")
    ax[2,2].imshow(og_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
    ax[2,2].imshow(ones, alpha = metrics["Mean Mask Applied"][isamp], cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ### COLUMN 4 ###
    ax[0,3].set_title("Infrared Image")
    ax[0,3].imshow(ir_image, cmap = "gray", origin = "lower")

    ax[1,3].set_title("Infrared Mask")
    ax[1,3].imshow(ir_image, cmap = "gray", origin = "lower")
    ax[1,3].imshow(metrics["IR Mask"][isamp], cmap = grn_cm, origin = 'lower') #, extent = (0,256,0,256))
    ax[1,3].set_xlabel("Threshold Value: 250K")

    ax[2,3].set_visible(False)

    ### COLUMN 5 ###
    ax[0,4].set_visible(False)

    ax[1,4].set_title("IR Mask Applied")
    ax[1,4].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[1,4].imshow(ones, alpha = metrics["IR Mask Applied Min"][isamp], cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,4].set_title("IR Mask Applied")
    ax[2,4].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[2,4].imshow(ones, alpha = metrics["IR Mask Applied Mean"][isamp], cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ### COLUMN 6 ###
    ax[0,5].set_title("Convolved Image")
    ax[0,5].imshow(cv_image, cmap = 'gray', origin = 'lower')

    ax[1,5].set_title("Convolve GLCM")
    ax[1,5].imshow(og_image, cmap = 'gray', origin = 'lower')
    #ax[1,5].set_xlabel("Max: " + "%.2f" % max(np.array(metrics["Max/Min Contrast C"])[:,0]) + "\n Min: " + "%.2f" % min(np.array(metrics["Max/Min Contrast C"])[:,1]))
    ax[1,5].imshow(ones, alpha = metrics["Convolved Contrast Values"][isamp], cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    ax[2,5].set_visible(False)

    ### COLUMN 7 ###
    ax[0,6].set_visible(False)

    ax[1,6].set_title("Min Mask Applied")
    ax[1,6].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[1,6].imshow(ones, alpha = metrics["Min Mask Applied C"][isamp], cmap = red_cm, origin = 'lower',extent = (0,256,0,256))

    ax[2,6].set_title("Mean Mask Applied")
    ax[2,6].imshow(og_image, cmap = 'gray', origin = 'lower')
    ax[2,6].imshow(ones, alpha = metrics["Mean Mask Applied C"][isamp], cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    for i in range (0,3):
        for j in range (0,7):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

#    plt.show()
    #plt.savefig()
    filepath = r'/home/nmitchell/GLCM/Images-Contrast/'
    filepath += 'Contrast_' + str(isamp) + ".png"
    plt.savefig(filepath)
    print(filepath)
    plt.close()
