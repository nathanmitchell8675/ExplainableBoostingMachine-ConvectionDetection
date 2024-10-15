from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

import numpy as np
import skimage as ski
import cv2 as cv
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xarray as xr
import pickle as pkl

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from imblearn.over_sampling import ADASYN

#Load the Training Data
training_data   = xr.open_dataset('/home/nmitchell/GLCM/testing_data.nc')

#Set the tile size
tile_size = 4

#Bring in the X variables -- Convolved Images, GLCM Values, and Infrared Images
### TRAINING DATA ###
#convolved_data_training  = training_data.Convolved_Image.values.flatten()
#infrared_data_training   = training_data.Infrared_Image.values.flatten()
#glcm_data_training_warm  = training_data.Above_IR_Mask_Applied_to_OG_GLCM.values.flatten()
#glcm_data_training_cool  = training_data.Below_IR_Mask_Applied_to_OG_GLCM.values.flatten()

#X_train = np.transpose(np.array([convolved_data_training, glcm_data_training_warm, glcm_data_training_cool, infrared_data_training]))
y_train = training_data.Masked_Truth.values.flatten()

print(sum(y_train)/len(y_train))

#print("Average Infrared Value:  ", np.nanmean(infrared_data_training))
#print("Average Infrared Value when Cool GLCM = 0: ", np.nanmean(np.where(glcm_data_training_cool == 0.0, np.nan, infrared_data_training)))
#print("Average Cool GLCM Value: ",  np.round(np.nanmean(np.where(glcm_data_training_cool != 0.0, glcm_data_training_cool, np.nan)),5))

#Oversample the data using ADASYN
#print("Oversampling the Data:")
#adasyn = ADASYN(random_state = 463, sampling_strategy = 0.03)
#X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

#print(np.sum(y_train)/len(y_train))

#Fits the model
#ebm = ExplainableBoostingClassifier(n_jobs = -1)
#print("Fitting the Model:")
#ebm.fit(X_train_res, y_train_res)

#print(ebm.eval_terms(X_train))
#print(ebm.predict_proba(X_train))
#y_pred = ebm.predict(X_val)
#y_pred = [float(i) for i in y_pred]
#print(metrics.confusion_matrix(y_val, y_pred))

#Saves the model in a dictionary
#model_dict = {"Model": []}
#model_dict["Model"].append(ebm)
#Pickles the Model
#filepath = r'/home/nmitchell/GLCM/models/EBM_model_ADASYN'
#pkl.dump(model_dict, open(filepath, 'wb'))
