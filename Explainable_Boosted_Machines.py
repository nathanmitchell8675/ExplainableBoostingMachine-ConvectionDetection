from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import axes
import skimage as ski
import cv2 as cv
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
import xarray as xr
import pickle as pkl

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from imblearn.over_sampling import ADASYN

#Load the Training Data
training_data   = xr.open_dataset('/home/nmitchell/GLCM/training_data.nc')

#Set the tile size
tile_size = 4

#Bring in the X variables -- Convolved Images, GLCM Values, and Infrared Images
### TRAINING DATA ###
convolved_data_training  = preprocessing.normalize(training_data.Convolved_Image.values, axis = 0).flatten()
glcm_data_training_ab    = preprocessing.normalize(training_data.Above_IR_Mask_Applied_to_OG_GLCM.values, axis = 0).flatten()
glcm_data_training_bl    = preprocessing.normalize(training_data.Below_IR_Mask_Applied_to_OG_GLCM.values, axis = 0).flatten()
infrared_data_training   = preprocessing.normalize(training_data.Infrared_Image.values, axis = 0).flatten()

#Bring in the y variable -- Expanded Ground Truths
exp_truth_training       = training_data.Expanded_Ground_Truth.values.flatten()
#ground_truth_training   = np.array(training_data.Ground_Truth).flatten().reshape(6033,64,64,4)[:,:,:,3].flatten()

#Concatenate the X variabes together to create X and rename the columns
X_train = np.transpose(pd.DataFrame((convolved_data_training, glcm_data_training_ab, glcm_data_training_bl, infrared_data_training),
                                     index = ["Convolved Data", "GLCM Above", "GLCM Below", "Infrared Data"]))

#Set y to be the Expanded Ground Truth
y_train = exp_truth_training

adasyn = ADASYN(random_state = 463)

X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

print("Fitting the Model:")

ebm = ExplainableBoostingClassifier(interactions = 0, n_jobs = -1)
ebm.fit(X_train_res, y_train_res)

#print(ebm.eval_terms(X_train))
#print(ebm.predict_proba(X_train))
#y_pred = ebm.predict(X_val)
#y_pred = [float(i) for i in y_pred]
#print(metrics.confusion_matrix(y_val, y_pred))

model_dict = {"Model": []}

model_dict["Model"].append(ebm)

#Pickles the Model
filepath = r'/home/nmitchell/GLCM/models/EBM_model'

pkl.dump(model_dict, open(filepath, 'wb'))

