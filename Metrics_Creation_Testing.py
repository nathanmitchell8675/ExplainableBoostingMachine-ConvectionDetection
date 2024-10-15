import numpy as np
import pandas as pd
import pickle as pkl
import cv2 as cv
import os
from skimage.feature import graycomatrix, graycoprops
import xarray as xr
#
from scipy.ndimage import uniform_filter
from skimage.restoration import inpaint
from scipy.ndimage import laplace
from scipy.ndimage import sobel
import copy
from collections import defaultdict
from itertools import groupby


# Input number of convection files and tile size
num  = 7786
tile_size = 4
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)


def fix_striping(data,a,b,c,d,e,first_pass):
    amap = data
    sav = copy.deepcopy(amap)
    amap = (laplace(amap, mode='mirror')).astype(int)
    amap = np.where(amap >= 200, amap - 256, amap)
    dx = sobel(amap, axis=0, mode='mirror')
    dy = sobel(amap, axis=1, mode='mirror')
    amap = np.degrees(np.arctan2(dy,dx))
    amap = (abs(amap) < 45).astype(int)

    ny,nx = amap.shape
    wx = 2.5
    fmap = np.empty(amap.shape)
    fmap.fill(-1)
    for iy,ix in np.ndindex(amap.shape):
        i1 = ix-(wx/2) if (ix-(wx/2) >= 0) else 0
        i2 = ix+(wx/2) if (ix+(wx/2) < nx) else nx-1
        section = amap[iy,int(i1):int(i2+1)]
        frac = np.sum(section)/len(section)
        if frac >= 0.99:
            fmap[iy,ix] = 1
            if (iy>d): fmap[iy-(a):iy+(a),     ix-(d):ix+(e)] = 1
            if (iy<(ny-a)): fmap[iy-(b):iy+(c),ix-(d):ix+(e)] = 1

    amap = np.where(fmap == 1, 1, sav)
    amap = np.expand_dims(amap, axis=2)
    smap = inpaint.inpaint_biharmonic(amap, (fmap==1), channel_axis=-1)
    smap = np.squeeze(smap)

    test_list = np.squeeze(np.where(amap == 1, 1, 0))
    max_lengths = np.zeros(256)
    max_percent = np.zeros(256)
    for i in range(test_list.shape[0]):
        max_percent[i] = np.sum(test_list[i,:])/256
        counter = defaultdict(list)
        new_list = test_list[i]
        if(np.sum(new_list) == 0):
            max_lengths[i] = 0
        else:
            for key, val in groupby(new_list, lambda ele: "one" if ele > 0 else "zero"):
                counter[key].append(len(list(val)))
            max_lengths[i] = max(counter['one'])

    max_l = max(max_lengths.flatten())
    max_p = max(max_percent.flatten())

    if (first_pass):
        return max_l, max_p
    else:
        interval_min = 0
        interval_max = 255
        scaled_mat = (smap - np.min(smap)) / (np.max(smap) - np.min(smap)) * (interval_max - interval_min) + interval_min
        return cv.blur(scaled_mat.astype(int), (3,3))

#Load in the Images:
images = xr.open_dataset('/home/nmitchell/GLCM/all_images.nc')

means       = []
mins        = []

metrics = {"Min Brightness": mins, "Mean Brightness": means}

samples = []

metrics_isamp = 0

num1 = 0
num2 = images.x_train_vis.sel(Sample = 20).shape[0]

file_num = 20

num_altered = 0

for n in range(num1, num2):
    isamp = n

    print("Sample Number: ", metrics_isamp)

    data  = np.array(images.x_train_vis.sel(Sample = file_num)[isamp,:,:,0])
    data *= 100
    data  = data.astype(np.uint8)

    if(np.sum(np.isnan(data)) > 0):
        continue

    data_truth  = np.array(images.y_train.sel(Sample = file_num)[isamp,:,:])
    data_truth *= 100
    data_truth  = data_truth.astype(np.uint8)
    truth_small = cv.resize(data_truth, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

    data_IR  = np.array(images.x_train_ir.sel(Sample = file_num)[isamp,:,:,0])

    #Gets rid of unnecessary values within the MRMS data
    data_truth  = np.where(data_truth == 100, 1, 0)
    truth_small = np.where(truth_small == 100, 1, 0)

    kernel_9x9     = np.ones((9,9), np.float32)/81
    convolve_data  = cv.filter2D(src = data, ddepth = -1, kernel = kernel_9x9)
    convolve_small = cv.resize(convolve_data, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)
    resized_IR     = cv.resize(data_IR.astype('float32'), (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

    #Determine if there is any striping present
#    max_l, max_p = fix_striping(data,1,-2,2,0,1,True)
#    if((max_l >= 15) | (max_p >= .25)):
    data = fix_striping(data,3,3,3,2,2,False)
    num_altered+=1

    #Put image data into the xarray format
    data_flat        = np.expand_dims(data.flatten(), 0)
    original_image   = xr.DataArray(data_flat, dims = ["Sample", "Length_256"], coords = {'Sample': np.ones(data_flat.shape[0])*metrics_isamp})

    truth_small      = np.expand_dims(truth_small.flatten(), 0)
    ground_truth     = xr.DataArray(truth_small, dims = ["Sample", "Length_64"], coords = {'Sample': np.ones(truth_small.shape[0])*metrics_isamp})

    data_convolved   = np.expand_dims(convolve_small.flatten(), 0)
    convolved_im     = xr.DataArray(data_convolved, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(data_convolved.shape[0])*metrics_isamp})

    resized_IR       = np.expand_dims(resized_IR.flatten(), 0)
    infrared_image   = xr.DataArray(resized_IR, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(resized_IR.shape[0])*metrics_isamp})

    full_truth       = np.expand_dims(data_truth.flatten(), 0)
    full_sized_mrms  = xr.DataArray(full_truth, dims = ["Sample", "Length_256"], coords = {"Sample": np.ones(full_truth.shape[0])*metrics_isamp})

    glcms            = []
    contrast_values  = []

    MMM = 0

    for r in range(0, 256, tile_size):
        for c in range(0, 256, tile_size):
            tile          = data[r:r+tile_size, c:c+tile_size]

            distances = [1]
            angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]

            glcm0 = graycomatrix(tile, distances = distances, angles=[0],           levels=256, symmetric = False)
            glcm1 = graycomatrix(tile, distances = distances, angles=[np.pi/4],     levels=256, symmetric = False)
            glcm2 = graycomatrix(tile, distances = distances, angles=[np.pi/2],     levels=256, symmetric = False)
            glcm3 = graycomatrix(tile, distances = distances, angles=[3 * np.pi/4], levels=256, symmetric = False)
            glcm=(glcm0 + glcm1 + glcm2 + glcm3)/4 #compute mean matrix
            glcms.append(glcm)

            contrast_values.append((float(graycoprops(glcm, 'contrast').ravel()[0])))

    #Normalize Contrast Values
    cvs = []
    for val in contrast_values:
        cvs.append(val/max(contrast_values))

    contrast_values = np.expand_dims(np.array(cvs), 0)
    GLCMs           = xr.DataArray(contrast_values, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(contrast_values.shape[0])*metrics_isamp})

    #IR MASK
    ir_vals_blw = (np.array((resized_IR <= 250).astype(int))).flatten()
    ir_vals_abv = (np.array((resized_IR > 250).astype(int))).flatten()

    ir_msk_blw  = np.expand_dims(np.where(ir_vals_blw, 1, np.nan), 0)
    ir_mk_abv   = np.expand_dims(np.where(ir_vals_abv, 1, np.nan), 0)
    ir_mask     = xr.DataArray(ir_msk_blw, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_msk_blw.shape[0])*metrics_isamp})

    ir_blw_     = np.multiply(ir_vals_blw, contrast_values)
    ir_abv_     = np.multiply(ir_vals_abv, contrast_values)
    ir_app_blw  = xr.DataArray(ir_blw_, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_blw_.shape[0])*metrics_isamp})
    ir_app_abv  = xr.DataArray(ir_abv_, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_abv_.shape[0])*metrics_isamp})

    #MRMS MASK
    mrms_blw  = np.multiply(ir_vals_blw, truth_small)
    mrms_mask = xr.DataArray(mrms_blw, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(mrms_blw.shape[0])*metrics_isamp})

    metrics_isamp = metrics_isamp + 1

    sample = xr.Dataset(data_vars = {"Original_Image": original_image, "Ground_Truth": ground_truth, "Convolved_Image": convolved_im,
                                     "Infrared_Image": infrared_image, "Masked_Truth": mrms_mask, "Original_GLCM": GLCMs, #"Expanded_Ground_Truth":
                                     "Above_IR_Mask_Applied_to_OG_GLCM": ir_app_abv, "Below_IR_Mask_Applied_to_OG_GLCM": ir_app_blw,
                                     "Full_Sized_MRMS": full_sized_mrms})

    samples.append(sample)

print("Num Altered: ", num_altered)

testing_data = xr.concat(samples, dim = "Sample")

testing_data.to_netcdf('/home/nmitchell/GLCM/testing_data_fixStripes_all.nc')
