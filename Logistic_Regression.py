import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import axes
import skimage as ski
import cv2 as cv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
import xarray as xr
import pickle as pkl
#####################################

#Load Training, Validation, & Testing Data
training_data   = xr.open_dataset('/home/nmitchell/GLCM/training_data.nc')
validation_data = xr.open_dataset('/home/nmitchell/GLCM/validation_data.nc')
#testing_data    = xr.open_dataset('/home/nmitchell/GLCM/testing_data.nc')

tile_size = 4

#Create the values that will become X -- the Convolved Image, the GLCM Values, and the Infrared Image
### TRAINING DATA ###
convolved_data_training   = preprocessing.normalize(training_data.Convolved_Image.values, axis = 0).flatten()
glcm_data_training        = preprocessing.normalize(training_data.Infrared_Mask_Applied_to_Min_Mask.values, axis = 0).flatten()
infrared_data_training    = preprocessing.normalize(training_data.Infrared_Image.values, axis = 0).flatten()

### VALIDATION DATA ###
convolved_data_validation = preprocessing.normalize(validation_data.Convolved_Image.values, axis = 0).flatten()
glcm_data_validation      = preprocessing.normalize(validation_data.Infrared_Mask_Applied_to_Min_Mask.values, axis = 0).flatten()
infrared_data_validation  = preprocessing.normalize(validation_data.Infrared_Image.values, axis = 0).flatten()

### TESTING DATA ###
#convolved_data_testing    = testing_data.Convolved_Image.values.flatten()
#glcm_data_testing         = testing_data.Original_GLCM.values.flatten()
#infrared_data_testing     = testing_data.Infrared_Image.values.flatten()


#Create 2 y variables -- the Ground Truth and the Expanded Ground Truth
### TRAINING DATA ###
#ground_truth_training   = np.array(training_data.Ground_Truth).flatten().reshape(6033,64,64,4)[:,:,:,3].flatten()
exp_truth_training      = training_data.Expanded_Ground_Truth.values.flatten()

### VALIDATION DATA ###
#ground_truth_validation = np.array(validation_data.Ground_Truth).flatten().reshape(902,64,64,4)[:,:,:,3].flatten()
exp_truth_validation    = validation_data.Expanded_Ground_Truth.values.flatten()

### TESTING DATA ###
#ground_truth_testing    = np.array(testing_data.Ground_Truth).flatten().reshape(851,64,64,4)[:,:,:,3].flatten()
#exp_truth_testing       = np.array(testing_data.Expanded_Ground_Truth).flatten().reshape(851,64,64,4)[:,:,:,3].flatten()


#Concatenate the X variabes together to create X and renames the columns
X_train = np.transpose(pd.DataFrame((convolved_data_training, glcm_data_training, infrared_data_training),
                                     #convolved_data_training*glcm_data_training,  infrared_data_training*glcm_data_training),
                                     index = ["Convolved Data", "GLCM Data", "Infrared Data"])) #, "Brightness/GLCM Interaction", "Infrared/GLCM Interaction"]))

X_val   = np.transpose(pd.DataFrame((convolved_data_validation, glcm_data_validation, infrared_data_validation),
                                     #convolved_data_validation*glcm_data_validation,  infrared_data_validation*glcm_data_validation),
                                     index = ["Convolved Data", "GLCM Data", "Infrared Data"])) #, "Brightness/GLCM Interaction", "Infrared/GLCM Interaction"]))

#X_test  = np.transpose(pd.DataFrame((convolved_data_testing, glcm_data_testing, infrared_data_testing), index = ["Convolved Data", "GLCM Data", "Infrared Data"]))

#Set y to be the Expanded Ground Truth and names the column
y_train = exp_truth_training
y_val   = exp_truth_validation
#y_test  = exp_truth_testing


print("Fitting the Models:")

unbalanced_weights = [{0: (np.bincount(y_train.astype(int))[0])/len(y_train), 1: (np.bincount(y_train.astype(int))[1])/len(y_train)}]
balanced_weights   = [{0: (len(y_train)/np.bincount(y_train.astype(int))[0]), 1: (len(y_train)/np.bincount(y_train.astype(int))[1])}]
#class_weights      = [{0: balanced_weights[0][0], 1: (1/5)*(balanced_weights[0][1] - unbalanced_weights[0][1])*i} for i in [1,2,3,4,5]]

#for i in range(len(class_weights)):
#    unbalanced_weights.append(class_weights[i])

weight = ((1/5)*(balanced_weights[0][1] - unbalanced_weights[0][1]))

class_weights = [{0: np.repeat(unbalanced_weights[0][0], 6)[i], 1: np.linspace(weight, weight*(1/6), 6)[i]} for i in [5,4,3,2,1,0]]

model_dict = {"Models": [], "Class Weights": class_weights, "Confusion Matrices": []}

for i in range(len(class_weights)):
    log_reg = LogisticRegression(class_weight = class_weights[i])
    log_reg.fit(X_train, y_train)
    model_dict["Models"].append(log_reg)

    y_pred = log_reg.predict(X_val)
    model_dict["Confusion Matrices"].append(metrics.confusion_matrix(y_val, y_pred))

#Pickles the Model
filepath = r'/home/nmitchell/GLCM/models/log_reg_models'

pkl.dump(model_dict, open(filepath, 'wb'))
