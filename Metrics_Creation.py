import numpy as np
import pandas as pd
import pickle as pkl
import cv2 as cv
import os
from skimage.feature import graycomatrix, graycoprops

# Input number of convection files and tile size
num  = 7786
tile_size = 4
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

#Load in the Images:
filepath = r'/home/nmitchell/GLCM/all_images'
with open(filepath, 'rb') as file:
    images = pkl.load(file)

og_image    = []
true_color  = []
conv_image  = []
IR_image    = []
means       = []
mins        = []
contrasts   = []
contrasts_c = []
mean_mask   = []
min_mask    = []
IR_mask     = []
mean_app    = []
mean_app_c  = []
min_app     = []
min_app_c   = []
ir_app_mean = []
ir_app_min  = []
maxmin      = []
maxmin_c    = []
expanded_gt = []

#flat_og     = []
flat_conv   = []
flat_glcm   = []
flat_truth  = []
#flat_min    = []
#flat_mean   = []
#flat_min_ap = []
#flat_mean_ap= []
flat_IR     = []
#flat_IR_min = []
#flat_IR_mean= []
flat_expand = []

metrics = {"Original Image": og_image, "Ground Truth": true_color, "Convolved Image": conv_image,
          "Infrared Image": IR_image, "Mean Brightness": means, "Min Brightness": mins, "Contrast Values": contrasts,
          "Convolved Contrast Values": contrasts_c, "Tile Mean Mask": mean_mask, "Tile Min Mask": min_mask,
          "IR Mask": IR_mask, "Mean Mask Applied": mean_app, "Min Mask Applied": min_app, "Mean Mask Applied C": mean_app_c,
          "Min Mask Applied C": min_app_c, "IR Mask Applied Mean": ir_app_mean, "IR Mask Applied Min": ir_app_min,
          "Max/Min Contrast": maxmin, "Max/Min Contrast C": maxmin_c, "Expanded Ground Truth": expanded_gt}

#flat_metrics = {"Flat Original Image": flat_og, "Flat Convolved Image": flat_conv, "Flat GLCM": flat_glcm, "Flat Ground Truth": flat_truth,
#                "Flat Min Mask": flat_min, "Flat Mean Mask": flat_mean, "Flat Applied Min Mask": flat_min_ap,
#                "Flat Applied Mean Mask": flat_mean_ap, "Flat Infrared Image": flat_IR, "Flat IR Mask Applied Min": flat_IR_min,
#                "Flat IR Mask Applied Mean": flat_IR_mean, "Flat Expanded Ground Truth": flat_expand}

flat_metrics = {"Flat Convolved Image": flat_conv, "Flat GLCM": flat_glcm, "Flat Ground Truth": flat_truth,
                "Flat Expanded Ground Truth": flat_expand, "Flat Infrared Image": flat_IR}

metrics_isamp = 0

for file_num in range(0, len(images["File Name"]) - 4):
    num1 = 0
    num2 = images["Data Length"][file_num]

    for n in range(num1, num2):
        isamp = n
        data  = images["X Train Vis"][file_num][isamp,:,:,0]
        data *= 100
        data  = data.astype(np.uint8)

        data_truth  = images["Y Train"][file_num][isamp,:,:]
        data_truth *= 100
        data_truth  = data_truth.astype(np.uint8)
        truth_small = cv.resize(data_truth, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

        data_IR  = images["X Train IR"][file_num][isamp,:,:,0]

        data_truth_color = np.zeros(len(truth_small)*len(truth_small)*4)
        data_truth_color = data_truth_color.reshape(len(truth_small),len(truth_small),4)

        expanded_gt = np.zeros((len(truth_small),len(truth_small),4))

        for i in range(len(truth_small)):
            for j in range(len(truth_small)):
                if(truth_small[i,j] == 100):
                    data_truth_color[i,j,:] = [1,0,0,1]

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
                    expanded_gt[a:b, c:d, :] = [1,0,0,1]

        kernel_9x9 = np.ones((9,9), np.float32)/81
        convolve_data  = cv.filter2D(src = data, ddepth = -1, kernel = kernel_9x9)
        convolve_small = cv.resize(convolve_data, (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)
        resized_IR = cv.resize(data_IR.astype('float32'), (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_NEAREST)

        metrics['Original Image'].append(data)
        metrics['Ground Truth'].append(data_truth_color)
        metrics['Convolved Image'].append(convolve_data)
        metrics['Infrared Image'].append(resized_IR)
        metrics['Mean Brightness'].append(np.mean(data))
        metrics['Min Brightness'].append(np.min(data))
        metrics["Expanded Ground Truth"].append(expanded_gt)

#       flat_metrics["Flat Original Image"].append(data.flatten())
        flat_metrics["Flat Convolved Image"].append(convolve_small.flatten())
        flat_metrics["Flat Ground Truth"].append((data_truth_color[:,:,3]).flatten())
        flat_metrics["Flat Expanded Ground Truth"].append((expanded_gt[:,:,3]).flatten())
        flat_metrics["Flat Infrared Image"].append(resized_IR.flatten())

        convolve_tiles    = []

        mean_tiles        = []
        min_tiles         = []

        glcms             = []
        glcms_c           = []

        contrast_values   = []
        contrast_values_c = []

        MMM = 0

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

        metrics["Contrast Values"].append(contrast_values)
        metrics["Convolved Contrast Values"].append(contrast_values_c)

        print("Sample Number: ", metrics_isamp)

        metrics["Max/Min Contrast"].append(np.array([max(contrast_values), min(contrast_values)]))
        metrics["Max/Min Contrast C"].append(np.array([max(contrast_values_c), min(contrast_values_c)]))

        max_con   = max(np.array(metrics["Max/Min Contrast"])[:,0])
        min_con   = min(np.array(metrics["Max/Min Contrast"])[:,1])
        max_con_c = max(np.array(metrics["Max/Min Contrast C"])[:,0])
        min_con_c = min(np.array(metrics["Max/Min Contrast C"])[:,1])

        metrics["Tile Min Mask"].append(np.array(min_tiles).reshape((int(256/tile_size), int(256/tile_size))))
        metrics["Tile Mean Mask"].append(np.array(mean_tiles).reshape((int(256/tile_size), int(256/tile_size))))

#       flat_metrics["Flat Min Mask"].append(metrics["Tile Min Mask"][isamp].flatten())
#       flat_metrics["Flat Mean Mask"].append(metrics["Tile Mean Mask"][isamp].flatten())


#    for n in range(num1, num2):
#        isamp = n

        cvs  = []
        cvsc = []

        for val in metrics["Contrast Values"][metrics_isamp]:
            cvs.append(val/max(metrics["Contrast Values"][metrics_isamp]))

        for val in metrics["Convolved Contrast Values"][metrics_isamp]:
            cvsc.append(val/max(metrics["Convolved Contrast Values"][metrics_isamp]))

        metrics["Contrast Values"][metrics_isamp] = np.array(cvs).reshape(int(256/tile_size), int(256/tile_size))
        metrics["Convolved Contrast Values"][metrics_isamp] = np.array(cvsc).reshape(int(256/tile_size), int(256/tile_size))

        flat_metrics["Flat GLCM"].append(metrics["Contrast Values"][metrics_isamp].flatten())

        #MIN MASK
        min_mask  = (metrics['Tile Min Mask'][metrics_isamp] >= 55.6125).astype(int)
        metrics["Tile Min Mask"][metrics_isamp]  = np.where(min_mask, 1, np.nan)
        metrics["Min Mask Applied"].append(np.multiply(metrics["Contrast Values"][metrics_isamp], min_mask))
        metrics["Min Mask Applied C"].append(np.multiply(metrics["Convolved Contrast Values"][metrics_isamp], min_mask))
#       flat_metrics["Flat Applied Min Mask"].append(metrics["Min Mask Applied"][isamp].flatten())

        #MEAN MASK
        mean_mask = (metrics["Tile Mean Mask"][metrics_isamp] >= 55.6125).astype(int)
        metrics["Tile Mean Mask"][metrics_isamp] = np.where(mean_mask, 1, np.nan)
        metrics["Mean Mask Applied"].append(np.multiply(metrics["Contrast Values"][metrics_isamp], mean_mask))
        metrics["Mean Mask Applied C"].append(np.multiply(metrics["Convolved Contrast Values"][metrics_isamp], mean_mask))
#       flat_metrics["Flat Applied Mean Mask"].append(metrics["Mean Mask Applied"][isamp].flatten())

        #IR MASK
        ir_mask   = (metrics["Infrared Image"][metrics_isamp] <= 250).astype(int)
        metrics["IR Mask"].append(np.where(ir_mask, 1, np.nan))
        metrics["IR Mask Applied Min"].append(np.multiply(metrics["Min Mask Applied"][metrics_isamp], ir_mask))
        metrics["IR Mask Applied Mean"].append(np.multiply(metrics["Mean Mask Applied"][metrics_isamp], ir_mask))
#       flat_metrics["Flat IR Mask Applied Min"].append(metrics["IR Mask Applied Min"][isamp].flatten())
#       flat_metrics["Flat IR Mask Applied Mean"].append(metrics["Ir Mask Applied Mean"][isamp].flatten())

        metrics_isamp = metrics_isamp + 1

filepath  = r'/home/nmitchell/GLCM/'
flat_path = r'/home/nmitchell/GLCM/'

filepath+= 'metrics_training'
flat_path+= 'flat_metrics_training'

pkl.dump(metrics, open(filepath, 'wb'))
pkl.dump(flat_metrics, open(flat_path, 'wb'))

