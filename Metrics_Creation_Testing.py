import numpy as np
import pandas as pd
import pickle as pkl
import cv2 as cv
import os
from skimage.feature import graycomatrix, graycoprops
import xarray as xr

# Input number of convection files and tile size
num  = 7786
tile_size = 4
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

#Load in the Images:
images = xr.open_dataset('/home/nmitchell/GLCM/all_images.nc')

means       = []
mins        = []

metrics = {"Min Brightness": mins, "Mean Brightness": means}

samples = []

metrics_isamp = 0

num1 = 0
num2 = images.x_train_vis.sel(Sample = 20).shape[0]

for n in range(num1, num2):
    print("Sample Number: ", metrics_isamp)
    metrics_isamp = metrics_isamp + 1

    isamp = n

    data  = np.array(images.x_train_vis.sel(Sample = 20)[isamp,:,:,0])
    data *= 100
    data  = data.astype(np.uint8)

    data_truth  = np.array(images.y_train.sel(Sample = 20)[isamp,:,:])
    data_truth *= 100
    data_truth  = data_truth.astype(np.uint8)
    truth_small = cv.resize(data_truth, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

    data_IR = np.array(images.x_train_ir.sel(Sample = 20)[isamp,:,:,0])

    expanded_gt = np.zeros((len(truth_small),len(truth_small)))

    for i in range(len(truth_small)):
        for j in range(len(truth_small)):
            if(truth_small[i,j] == 100):
                truth_small[i,j] = 1
                if((i-1) < 0):
                    a = 0
                else:
                    a = (i-1)
                if((i+2) > len(truth_small)):
                    b = len(truth_small)
                else:
                    b = (i+2)
                if((j-1) < 0):
                    c = 0
                else:
                    c = (j-1)
                if((j+2) > len(truth_small)):
                    d = len(truth_small)
                else:
                    d = (j+2)
                expanded_gt[a:b, c:d] = 1
            else:
                truth_small[i,j] = 0

    kernel_9x9     = np.ones((9,9), np.float32)/81
    convolve_data  = cv.filter2D(src = data, ddepth = -1, kernel = kernel_9x9)
    convolve_small = cv.resize(convolve_data, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)
    resized_IR     = cv.resize(data_IR.astype('float32'), (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

    #Put image data into the xarray format
    data_flat        = np.expand_dims(data.flatten(), 0)
    original_image   = xr.DataArray(data_flat, dims = ["Sample", "Length_256"], coords = {'Sample': np.ones(data_flat.shape[0])*isamp})

    truth_small      = np.expand_dims(truth_small.flatten(), 0)
    ground_truth     = xr.DataArray(truth_small, dims = ["Sample", "Length_64"], coords = {'Sample': np.ones(truth_small.shape[0])*isamp})

    data_convolved   = np.expand_dims(convolve_small.flatten(), 0)
    convolved_im     = xr.DataArray(data_convolved, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(data_convolved.shape[0])*isamp})

    resized_IR       = np.expand_dims(resized_IR.flatten(), 0)
    infrared_image   = xr.DataArray(resized_IR, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(resized_IR.shape[0])*isamp})

    expanded_gt      = np.expand_dims(expanded_gt.flatten(), 0)
    expanded_truth   = xr.DataArray(expanded_gt, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(expanded_gt.shape[0])*isamp})

    metrics['Mean Brightness'].append(np.mean(data))
    metrics['Min Brightness'].append(np.min(data))

    convolve_tiles    = []

    mean_tiles        = []
    min_tiles         = []

    glcms             = []
    glcms_c           = []

    contrast_values   = []
    contrast_values_c = []

    for r in range(0, 256, tile_size):
        for c in range(0, 256, tile_size):
            tile          = data[r:r+tile_size, c:c+tile_size]
            convolve_tile = convolve_data[r:r + tile_size, c:c + tile_size]

            convolve_tiles.append(convolve_tile)

            distances = [1]
            angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]

            glcm0 = graycomatrix(tile, distances = distances, angles=[0],           levels=256, symmetric = False)
            glcm1 = graycomatrix(tile, distances = distances, angles=[np.pi/4],     levels=256, symmetric = False)
            glcm2 = graycomatrix(tile, distances = distances, angles=[np.pi/2],     levels=256, symmetric = False)
            glcm3 = graycomatrix(tile, distances = distances, angles=[3 * np.pi/4], levels=256, symmetric = False)
            glcm=(glcm0 + glcm1 + glcm2 + glcm3)/4 #compute mean matrix
            glcms.append(glcm)

            #GLCM for Convolved Data
            glcm0_c = graycomatrix(convolve_tile, distances = distances, angles = [0],           levels = 256, symmetric = False)
            glcm1_c = graycomatrix(convolve_tile, distances = distances, angles = [np.pi/4],     levels = 256, symmetric = False)
            glcm2_c = graycomatrix(convolve_tile, distances = distances, angles = [np.pi/2],     levels = 256, symmetric = False)

            glcm3_c = graycomatrix(convolve_tile, distances = distances, angles = [3 * np.pi/4], levels = 256, symmetric = False)
            glcm_c  = (glcm0_c + glcm1_c + glcm2_c + glcm3_c)/4 #compute mean matrix
            glcms_c.append(glcm_c)

            contrast_values.append((float(graycoprops(glcm, 'contrast').ravel()[0])))

            contrast_values_c.append((float(graycoprops(glcm_c, 'contrast').ravel()[0])))

            mean_tiles.append(np.mean(convolve_tile))
            min_tiles.append(np.min(convolve_tile))

    #Normalize Contrast Values
    cvs = []
    for val in contrast_values:
        cvs.append(val/max(contrast_values))

    contrast_values = np.expand_dims(np.array(cvs), 0)
    GLCMs           = xr.DataArray(contrast_values, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(contrast_values.shape[0])*isamp})

    #Normalize Convolved Contrast Values
    cvsc = []
    for val in contrast_values_c:
         cvsc.append(val/max(contrast_values_c))

    contrast_vals_c = np.expand_dims(np.array(cvsc), 0)
    GLCMs_c         = xr.DataArray(contrast_vals_c, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(contrast_vals_c.shape[0])*isamp})

    #MIN MASK
    min_vals    = ((np.array(min_tiles).reshape((int(256/tile_size), int(256/tile_size))) >= 55.6125).astype(int)).flatten()

    min_msk     = np.expand_dims(np.where(min_vals, 1, np.nan), 0)
    min_mask    = xr.DataArray(min_msk, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(min_msk.shape[0])*isamp})

    min_app_    = np.multiply(min_vals, contrast_values)
    min_app     = xr.DataArray(min_app_, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(min_app_.shape[0])*isamp})

    min_appc    = np.multiply(min_vals, contrast_vals_c)
    min_app_c   = xr.DataArray(min_appc, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(min_appc.shape[0])*isamp})

    #MEAN MASK
    mean_vals   = ((np.array(mean_tiles).reshape((int(256/tile_size)), (int(256/tile_size))) >= 55.6125).astype(int)).flatten()

    mean_msk    = np.expand_dims(np.where(mean_vals, 1, np.nan), 0)
    mean_mask   = xr.DataArray(mean_msk, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(mean_msk.shape[0])*isamp})

    mean_app_   = np.multiply(mean_vals, contrast_values)
    mean_app    = xr.DataArray(mean_app_, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(mean_app_.shape[0])*isamp})

    mean_appc   = np.multiply(mean_vals, contrast_vals_c)
    mean_app_c  = xr.DataArray(mean_appc, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(mean_appc.shape[0])*isamp})

    #IR MASK
    ir_values   = (np.array((resized_IR <= 250).astype(int))).flatten()

    ir_msk      = np.expand_dims(np.where(ir_values, 1, np.nan), 0)
    ir_mask     = xr.DataArray(ir_msk, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_msk.shape[0])*isamp})

    ir_min      = np.multiply(min_app_, ir_values)
    ir_app_min  = xr.DataArray(ir_min, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_min.shape[0])*isamp})

    ir_mean     = np.multiply(mean_app_, ir_values)
    ir_app_mean = xr.DataArray(ir_mean, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_mean.shape[0])*isamp})

    sample = xr.Dataset(data_vars = {"Original_Image": original_image, "Ground_Truth": ground_truth, "Convolved_Image": convolved_im,
                                     "Infrared_Image": infrared_image, "Expanded_Ground_Truth": expanded_truth, "Original_GLCM": GLCMs,
                                     "Convolved_GLCM": GLCMs_c, "Min_Mask": min_mask, "Min_Mask_Applied": min_app,
                                     "Min_Mask_Applied_Convolved_Image": min_app_c, "Mean_Mask": mean_mask, "Mean_Mask_Applied": mean_app_c,
                                     "Mean_Mask_Applied_Convolved_Image": mean_app_c, "Infrared_Image_Mask": ir_mask,
                                     "Infrared_Mask_Applied_to_Min_Mask": ir_app_min, "Infrared_Mask_Applied_to_Mean_Mask": ir_app_mean})

    samples.append(sample)

testing_data = xr.concat(samples, dim = "Sample")

testing_data.to_netcdf('/home/nmitchell/GLCM/testing_data.nc')
