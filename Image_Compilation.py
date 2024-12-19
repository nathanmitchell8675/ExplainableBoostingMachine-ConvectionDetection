import numpy as np
import pandas as pd
import pickle as pkl
import os
import xarray as xr

num1 = 0
variables = []

with os.scandir('/mnt/data1/ylee/for_Jason') as entries:
    i = -1
    for entry in entries:
        if 'seg_mrms_256' in entry.name:
            i = i + 1

            filepath = '/mnt/data1/ylee/for_Jason/' + entry.name

            data_temp = np.fromfile((open(filepath, 'rb')), dtype = 'float32')

            temp_num = int(len(data_temp)/692224)

            x_train_vis = np.zeros((temp_num,256,256,9), dtype = 'float32')
            x_train_ir  = np.zeros((temp_num,64,64,9),   dtype = 'float32')
            y_train     = np.zeros((temp_num,256,256),   dtype = 'float32')

            for j in range(num1, temp_num):
                x_train_vis[j,:,:,:] = np.reshape(data_temp[(j*(692224)):(j*(692224)+589824)],(256,256,9))
                x_train_ir[j,:,:,:]  = np.reshape(data_temp[(589824+j*(692224)):(589824+j*(692224)+36864)],(64,64,9))
                y_train[j,:,:]       = np.reshape(data_temp[(626688 + j*(692224)):(626688+j*(692224)+65536)], (256,256))

            a = xr.DataArray(x_train_vis, dims = ["Sample", "X", "Y", "Z"],       coords = {'Sample': np.ones(x_train_vis.shape[0])*i})
            b = xr.DataArray(x_train_ir,  dims = ["Sample", "X_IR", "Y_IR", "Z"], coords = {'Sample': np.ones(x_train_ir.shape[0])*i})
            c = xr.DataArray(y_train,     dims = ["Sample", "X", "Y"],            coords = {'Sample': np.ones(y_train.shape[0])*i})

            variable = xr.Dataset(data_vars = {"x_train_vis": a, "x_train_ir": b, "y_train": c})
            variables.append(variable)

all_images = xr.concat(variables, dim = "Sample")
all_images.to_netcdf('/home/nmitchell/GLCM/all_images.nc')


