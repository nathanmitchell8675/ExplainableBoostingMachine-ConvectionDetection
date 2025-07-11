from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

import numpy as np
import xarray as xr
import pickle as pkl

from interpret.glassbox import ExplainableBoostingClassifier

import cv2 as cv

#Load the Training Data
#training_data   = xr.open_dataset('/home/nmitchell/GLCM/reflec_training.nc')
training_data   = xr.open_dataset('/home/nmitchell/GLCM/training1.nc')

#Brightness
brightness_data_training  = training_data.Brightness.values.flatten()

#Infrared
infrared_data_training   = training_data.Infrared_Image.values.flatten()

#Contrast Tiles (for getting value to normalize by)
tiles = training_data.Original_GLCM.values.flatten()
tiles = np.log(tiles+1)
max_value = np.max(tiles)

#Cool Contrast Tiles
glcm_cool  = np.log(training_data.Cool_Contrast_Tiles.values.flatten() + 1)
glcm_data_training_cool = np.where(glcm_cool <= max_value, glcm_cool/max_value, 1)

#Define the X matrix / y array
X_train = np.transpose(np.array([brightness_data_training, infrared_data_training, glcm_data_training_cool]))
y_train = training_data.Masked_Truth.values.flatten()

#Fit the EBM
ebm = ExplainableBoostingClassifier(n_jobs = -1, smoothing_rounds = 500, random_state = 1)
print("Fitting the Model:")
ebm.fit(X_train, y_train)

#Save the model in a dictionary
model_dict = {"Model": []}
model_dict["Model"].append(ebm)

#Pickle the dictionary
filepath = r'/home/nmitchell/GLCM/models/13BM'
pkl.dump(model_dict, open(filepath, 'wb'))
