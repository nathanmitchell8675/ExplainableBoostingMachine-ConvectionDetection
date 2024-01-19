import numpy as np
import pandas as pd
import pickle as pkl
import cv2 as cv
from skimage.feature import graycomatrix, graycoprops
# Input number of convection files and tile size
num= 204
num1 = 0
num2 = 204
tile_size = 8
num_rows = int(256/tile_size)
num_cols = int(256/tile_size)
num_tiles = int(num_rows * num_cols)

# Get sample image from Convection data
x_train_vis = np.zeros((num,256,256,9), dtype = 'float32')
x_train_ir  = np.zeros((num,64,64,9),   dtype = 'float32')
y_train     = np.zeros((num,256,256),   dtype = 'float32')
f = open(r'/mnt/data1/ylee/for_Jason/20190523_seg_mrms_256_comp_real.bin','rb')
data = np.fromfile(f,dtype='float32')

for j in range(num1, num2):
        x_train_vis[j,:,:,:] = np.reshape(data[(j*(692224)):(j*(692224)+589824)],(256,256,9))
        x_train_ir[j,:,:,:]  = np.reshape(data[(589824+j*(692224)):(589824+j*(692224)+36864)],(64,64,9))
        y_train[j,:,:]       = np.reshape(data[(626688 + j*(692224)):(626688+j*(692224)+65536)], (256,256))

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

metrics = {"Original Image": og_image, "Ground Truth": true_color, "Convolved Image": conv_image,
          "Infrared Image": IR_image, "Mean Brightness": means, "Min Brightness": mins, "Contrast Values": contrasts,
          "Convolved Contrast Values": contrasts_c, "Tile Mean Mask": mean_mask, "Tile Min Mask": min_mask,
          "IR Mask": IR_mask, "Mean Mask Applied": mean_app, "Min Mask Applied": min_app, "Mean Mask Applied C": mean_app_c,
          "Min Mask Applied C": min_app_c, "IR Mask Applied Mean": ir_app_mean, "IR Mask Applied Min": ir_app_min,
          "Max/Min Contrast": maxmin, "Max/Min Contrast C": maxmin_c}

for n in range(num1, num2):
    isamp = n
    data  = x_train_vis[isamp,:,:,0]
    data *= 100
    data  = data.astype(np.uint8)

    data_truth  = y_train[isamp,:,:]
    data_truth *= 100
    data_truth  = data_truth.astype(np.uint8)

    data_IR  = x_train_ir[isamp,:,:,0]

    data_truth_color = np.zeros(len(data_truth)*len(data_truth)*4)
    data_truth_color = data_truth_color.reshape(len(data_truth),len(data_truth),4)

    for i in range(len(data_truth)):
        for j in range(len(data_truth)):
            if(data_truth[i,j] == 100):
                data_truth_color[i,j,:] = [1,0,0,1]
            else:
                data_truth_color[i,j,:] = [1,1,1,1]


    kernel_9x9 = np.ones((9,9), np.float32)/81
    convolve_data = cv.filter2D(src = data, ddepth = -1, kernel = kernel_9x9)
    resized_IR = cv.resize(data_IR.astype('float32'), (int(256/tile_size), int(256/tile_size)), interpolation = cv.INTER_CUBIC)

    metrics['Original Image'].append(data)
    metrics['Ground Truth'].append(data_truth_color)
    metrics['Convolved Image'].append(convolve_data)
    metrics['Infrared Image'].append(resized_IR)
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

    metrics["Contrast Values"].append(contrast_values)
    metrics["Convolved Contrast Values"].append(contrast_values_c)

    metrics["Max/Min Contrast"].append(np.array([max(contrast_values), min(contrast_values)]))
    metrics["Max/Min Contrast C"].append(np.array([max(contrast_values_c), min(contrast_values_c)]))

    max_con   = max(np.array(metrics["Max/Min Contrast"])[:,0])
    min_con   = min(np.array(metrics["Max/Min Contrast"])[:,1])
    max_con_c = max(np.array(metrics["Max/Min Contrast C"])[:,0])
    min_con_c = min(np.array(metrics["Max/Min Contrast C"])[:,1])

    metrics["Tile Mean Mask"].append(np.array(mean_tiles).reshape((int(256/tile_size), int(256/tile_size))))
    metrics["Tile Min Mask"].append(np.array(min_tiles).reshape((int(256/tile_size), int(256/tile_size))))

for n in range(num1, num2):
    isamp = n

    cvs  = []
    cvsc = []

#    for val in metrics["Contrast Values"][isamp]:
#        cvs.append((val - min_con)/(max_con - min_con))

#    for val in metrics["Convolved Contrast Values"][isamp]:

#        cvsc.append((val - min_con_c)/(max_con_c - min_con_c))

    for val in metrics["Contrast Values"][isamp]:
        cvs.append(val/max(metrics["Contrast Values"][isamp]))

    for val in metrics["Convolved Contrast Values"][isamp]:
        cvsc.append(val/max(metrics["Convolved Contrast Values"][isamp]))

    metrics["Contrast Values"][isamp] = np.array(cvs).reshape(int(256/tile_size), int(256/tile_size))
    metrics["Convolved Contrast Values"][isamp] = np.array(cvsc).reshape(int(256/tile_size), int(256/tile_size))

    mean_mask = (metrics["Tile Mean Mask"][isamp] >= np.mean(metrics["Mean Brightness"])*1.5).astype(int)
    metrics["Tile Mean Mask"][isamp] = np.where(mean_mask, 1, np.nan)

    metrics["Mean Mask Applied"].append(np.multiply(metrics["Contrast Values"][isamp], mean_mask))
    metrics["Mean Mask Applied C"].append(np.multiply(metrics["Convolved Contrast Values"][isamp], mean_mask))

    min_mask  = (metrics["Tile Min Mask"][isamp] >= np.mean(metrics["Mean Brightness"])*1.5).astype(int)
    metrics["Tile Min Mask"][isamp]  = np.where(min_mask, 1, np.nan)
    metrics["Min Mask Applied"].append(np.multiply(metrics["Contrast Values"][isamp], min_mask))
    metrics["Min Mask Applied C"].append(np.multiply(metrics["Convolved Contrast Values"][isamp], min_mask))

    ir_mask   = (metrics["Infrared Image"][isamp] <= 250).astype(int)
    metrics["IR Mask"].append(np.where(ir_mask, 1, np.nan))
    metrics["IR Mask Applied Mean"].append(np.multiply(metrics["Mean Mask Applied"][isamp], ir_mask))
    metrics["IR Mask Applied Min"].append(np.multiply(metrics["Min Mask Applied"][isamp], ir_mask))

filepath = r'/home/nmitchell/GLCM/'
filepath+= 'metrics'
pkl.dump(metrics, open(filepath, 'wb'))

