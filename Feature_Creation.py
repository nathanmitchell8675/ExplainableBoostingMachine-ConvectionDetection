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

import matplotlib as mpl
import matplotlib.pyplot as plt

# Input number of convection files and tile size
tile_size = 4
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

#Load in the Images:
images = xr.open_dataset('/mnt/data1/hilburn/conus5/conus5.nc')

samples = []
metrics_isamp = 0

def possol(jday, tu, xlon, xlat):
    # input: jday = julian day
    #tu = time of day (fractional hours)
    #lon = longitude (deg)
    #lat = latitude (deg)
    #output: asol = solar zenith angle (deg)

    #constants
    pi = np.pi
    dtor = pi / 180.0

    #mean solar time
    tsm = tu + xlon/15.0
    xlo = xlon*dtor
    xla = xlat*dtor
    xj = float(jday)

    #time equation (mn.dec)
    a1 = (1.00554*xj - 6.28306) * dtor
    a2 = (1.93946*xj + 23.35089) * dtor
    et = -7.67825*np.sin(a1) - 10.09176*np.sin(a2)

    #true solar time
    tsv = tsm + et/60.0
    tsv = tsv - 12.0

    #hour angle
    ah = tsv*15.0*dtor

    #solar declination (in radians)
    a3 = (0.9683*xj - 78.00878) * dtor
    delta = 23.4856*np.sin(a3)*dtor

    #elevation
    amuzero = np.sin(xla)*np.sin(delta) + np.cos(xla)*np.cos(delta)*np.cos(ah)
    elev = np.arcsin(amuzero) #alpha_s, solar altitude angle (in radians)

    #conversion in degrees
    elev = elev / dtor #convert alpha_s to degrees
    asol = 90.0 - elev #theta_s = 90 - alpha_s

    return asol

#Load the Channel 2 reflectance data, get the min and max (for normalization later on)
max_02 = 2.3596193272540766
min_02 = -0.0014044149232572435

max_glcm = 7.052267824115893

#6,003 samples in the training dataset (2021 & 2022)
#2,939 samples in the validation dataset (2023)
#3,066 samples in the testing dataset (2024)

#Training data
#num1 = 0
#num2 = 6003

#Validation data
#num1 = 6003
#num2 = 8942

#Test data
num1 = 8942
num2 = 12008

for n in range(num1,num2):
    isamp = n

    print("Sample Number: ", metrics_isamp)

    #Get the Case Number -- used to get the lat/lon
    caseNum = images.case_number.sel(nsamp = isamp).values

    #Load reflectance data, normalize to 0 - 1, re-scale to 0 - 255
    reflectance_factor = images.C02.sel(nsamp = isamp).values

    #Load Latitude/Longitude (high res)
    lat_high = images.lat_hi.sel(ncases = caseNum).values
    lon_high = images.lon_hi.sel(ncases = caseNum).values

    t1 = pd.Timestamp(images.datetime.sel(nsamp = isamp).values)
    t = ((t1.hour)%24) + (t1.minute/60) + (t1.second/3600)
    d = float(t1.timetuple().tm_yday)
    rads = np.pi/180.0

    degree = possol(d, t, lon_high, lat_high)

    if(np.min(degree.flatten()) < 65):

        metrics_isamp = metrics_isamp + 1

        zenith_angle  = np.cos(degree*(np.pi/180))

        reflectance = reflectance_factor/zenith_angle
        #reflectance = ((reflectance - min_02)/(max_02 - min_02))*255

        truncated = np.where(reflectance >= min_02, reflectance - min_02, 0) / (max_02 - min_02)
        reflectance = np.where(truncated > 1, 1, truncated)*255

        #Define kernel for blurring reflectance data, resize to create the Brightness feature
        kernel_9x9      = np.ones((9,9), np.float32)/(9*9)
        blurred_reflec  = cv.filter2D(src = reflectance, ddepth = -1, kernel = kernel_9x9)
        brightness      = cv.resize(blurred_reflec, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

        #Load Infrared data to create the Infrared feature
        infra = images.C13.sel(nsamp = isamp).values.flatten()

        #Load MRMS data, resize
        mrms       = images.CFLAG.sel(nsamp = isamp).values
        mrms_small = cv.resize(mrms, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

        #Load Latitude/Longitude (low res)

        lat_low  = images.lat_lo.sel(ncases = caseNum).values.flatten()
        lon_low  = images.lon_lo.sel(ncases = caseNum).values.flatten()

        #Put data into the xarray format:

        date = xr.DataArray(t1, dims = ["Sample", "Length_1"], coords = {'Sample': np.ones(1)*metrics_isamp})

        #Latitude/Longitude Data (High- and Low-Res)
        lathi            = np.expand_dims(lat_high.flatten(), 0)
        latitude_high    = xr.DataArray(lathi, dims = ["Sample", "Length_256"], coords = {'Sample': np.ones(lathi.shape[0])*metrics_isamp})

        lonhi            = np.expand_dims(lon_high.flatten(), 0)
        longitude_high   = xr.DataArray(lonhi, dims = ["Sample", "Length_256"], coords = {'Sample': np.ones(lonhi.shape[0])*metrics_isamp})


        latlo            = np.expand_dims(lat_low.flatten(), 0)
        latitude_low     = xr.DataArray(latlo, dims = ["Sample", "Length_64"], coords = {'Sample': np.ones(latlo.shape[0])*metrics_isamp})

        lonlo            = np.expand_dims(lon_low.flatten(), 0)
        longitude_low    = xr.DataArray(lonlo, dims = ["Sample", "Length_64"], coords = {'Sample': np.ones(lonlo.shape[0])*metrics_isamp})

        #Reflectance Data ("Original Image")
        reflec_flat      = np.expand_dims(reflectance.flatten(), 0)
        original_image   = xr.DataArray(reflec_flat, dims = ["Sample", "Length_256"], coords = {'Sample': np.ones(reflec_flat.shape[0])*metrics_isamp})

        #Resized MRMS Data ("Ground Truth")
        truth_small      = np.expand_dims(mrms_small.flatten(), 0)
        ground_truth     = xr.DataArray(truth_small, dims = ["Sample", "Length_64"], coords = {'Sample': np.ones(truth_small.shape[0])*metrics_isamp})

        #Convolved Data ("Brightness")
        data_convolved   = np.expand_dims(brightness.flatten(), 0)
        brightness       = xr.DataArray(data_convolved, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(data_convolved.shape[0])*metrics_isamp})

        #Infrared Data ("Infrared Image")
        resized_IR       = np.expand_dims(infra.flatten(), 0)
        infrared_image   = xr.DataArray(resized_IR, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(resized_IR.shape[0])*metrics_isamp})

        #Full-Sized MRMS Data
        full_truth       = np.expand_dims(mrms.flatten(), 0)
        full_sized_mrms  = xr.DataArray(full_truth, dims = ["Sample", "Length_256"], coords = {"Sample": np.ones(full_truth.shape[0])*metrics_isamp})

        #Gray-Level Co-occurrence Matrices
        glcms            = []
        contrast_values  = []

        MMM = 0

        #Convert reflectance to integer values (GLCM tiles cannot be computed otherwise!)
        data = reflectance.astype(np.uint8).reshape(256,256)

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

        cvs = []
        for val in contrast_values:
            val  = np.log(val + 1)
            val  = np.where(val <= max_glcm, val/max_glcm, 1)
            cvs.append(val)

        contrast_values = np.expand_dims(np.array(cvs), 0)
        GLCMs           = xr.DataArray(contrast_values, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(contrast_values.shape[0])*metrics_isamp})

        #IR MASK
        ir_vals_blw = (np.array((resized_IR <= 250).astype(int))).flatten()
        ir_vals_abv = (np.array((resized_IR > 250).astype(int))).flatten()

        ir_blw_     = np.multiply(ir_vals_blw, contrast_values)
        ir_abv_     = np.multiply(ir_vals_abv, contrast_values)
        ir_app_blw  = xr.DataArray(ir_blw_, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_blw_.shape[0])*metrics_isamp})
        ir_app_abv  = xr.DataArray(ir_abv_, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(ir_abv_.shape[0])*metrics_isamp})

        #MRMS MASK
        mrms_blw  = np.multiply(ir_vals_blw, truth_small)
        mrms_mask = xr.DataArray(mrms_blw, dims = ["Sample", "Length_64"], coords = {"Sample": np.ones(mrms_blw.shape[0])*metrics_isamp})

        sample = xr.Dataset(data_vars = { "Original_Image": original_image, "Ground_Truth": ground_truth, "Brightness": brightness,
                                         "Infrared_Image": infrared_image, "Masked_Truth": mrms_mask, "Original_GLCM": GLCMs,
                                         "Warm_Contrast_Tiles": ir_app_abv, "Cool_Contrast_Tiles": ir_app_blw,
                                         "Full_Sized_MRMS": full_sized_mrms, "Latitude_High": latitude_high, "Longitude_High": longitude_high,
                                         "Latitude_Low": latitude_low, "Longitude_Low": longitude_low, "Date": date})

        samples.append(sample)

        metrics_isamp = metrics_isamp + 1

    else:
        print("Sample",isamp,"had a degree over 65")
        metrics_isamp = metrics_isamp + 1
        continue

testing_data = xr.concat(samples, dim = "Sample")

testing_data.to_netcdf('/home/nmitchell/GLCM/test1.nc')

