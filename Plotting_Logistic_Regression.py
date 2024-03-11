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
import xarray as xr
import pickle as pkl
#####################################

tile_size = 4

filepath = r'/home/nmitchell/GLCM/models/log_reg_models'
with open(filepath, 'rb') as file:
    models = pkl.load(file)

unbalanced_model = models["Models"][0]
second_model     = models["Models"][1]
third_model      = models["Models"][2]
fourth_model     = models["Models"][3]
fifth_model      = models["Models"][4]
balanced_model   = models["Models"][5]

validation_data = xr.open_dataset('/home/nmitchell/GLCM/validation_data.nc')

ones = np.ones((int(256/tile_size), int(254/tile_size)))
red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["red", "black"])
gt_cm  = mpl.colors.LinearSegmentedColormap.from_list(" ", ["white", "red"])

num1 = 1
num2 = 150

for isamp in range(num1,num2):
    #Pull the Ground Truth, the Expanded Ground Truth, the Convolved Image, the Original GLCM Tiles,
    #and the Infrared Image from the testing dataset. To change images, change the Sample number (isamp).
    val_gt    = validation_data.Ground_Truth.sel(Sample = isamp).values.flatten()
    val_egt   = validation_data.Expanded_Ground_Truth.sel(Sample = isamp).values.flatten()

    val_conv  = validation_data.Convolved_Image.sel(Sample = isamp).values.flatten()
    val_glcm  = validation_data.Original_GLCM.sel(Sample = isamp).values.flatten()
    val_ir    = validation_data.Infrared_Image.sel(Sample = isamp).values.flatten()

    X_val  = np.transpose(pd.DataFrame((val_conv, val_glcm, val_ir), index = ["Convolved Data", "GLCM Data", "Infrared Data"]))
    y_val  = val_egt

    #Predicts the GLCM values for the reserved testing image
    pred_ubal        = unbalanced_model.predict(X_val)
    pred_ubal_probs  = unbalanced_model.predict_proba(X_val)[:,1]

    pred_two         = second_model.predict(X_val)
    pred_two_probs   = second_model.predict_proba(X_val)[:,1]

    pred_three       = third_model.predict(X_val)
    pred_three_probs = third_model.predict_proba(X_val)[:,1]

    pred_four        = fourth_model.predict(X_val)
    pred_four_probs  = fourth_model.predict_proba(X_val)[:,1]

    pred_five        = fifth_model.predict(X_val)
    pred_five_probs  = fifth_model.predict_proba(X_val)[:,1]

    pred_bal         = balanced_model.predict(X_val)
    pred_bal_probs   = balanced_model.predict_proba(X_val)[:,1]


    fig, ax = plt.subplots(2,6)

    convolved_image  = val_conv.reshape(64,64)
    ground_truth     = val_gt.reshape(64,64)
    exp_ground_truth = val_egt.reshape(64,64)

    ### FIRST ROW ###
    ax[0,0].set_visible(False)

    ax[0,1].set_title("Convolved Image")
    ax[0,1].imshow(convolved_image, cmap = 'gray', origin = 'lower')

    ax[0,2].set_title("Ground Truth")
    ax[0,2].imshow(ground_truth, cmap = gt_cm, origin = 'lower')

    ax[0,3].set_title("Expanded Ground Truth")
    ax[0,3].imshow(exp_ground_truth, cmap = gt_cm, origin = 'lower')

    ax[0,4].set_title("Original GLCM Tiles")
    ax[0,4].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[0,4].imshow(ones, alpha = val_glcm.reshape(64,64), cmap = red_cm, origin = 'lower')

    ax[0,5].set_visible(False)

    ### SECOND ROW ###
    ax[1,0].set_title("Unbalanced Model\nPredicted GLCM Tiles\n" + 'Class Weights:\n' +
                       "0: " + str(round(models["Class Weights"][0][0],3)) + ", " +
                       "1: " + str(round(models["Class Weights"][0][1],3)))
    ax[1,0].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,0].imshow(ones, alpha = (pred_ubal*pred_ubal_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,0].set_xlabel("Weights:\n" +
                       "Convolved Image: " + str(round(unbalanced_model.coef_[0][0], 3)) + "\n" +
                       "GLCM Tiles:      " + str(round(unbalanced_model.coef_[0][1], 3)) + "\n" +
                       "Infrared Image:  " + str(round(unbalanced_model.coef_[0][2], 3)))

    ax[1,1].set_title("Class Weights:" + "\n" +
                             "0: " + str(round(models["Class Weights"][1][0],3)) + ", " +
                             "1: " + str(round(models["Class Weights"][1][1],3)))
    ax[1,1].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,1].imshow(ones, alpha = (pred_two*pred_two_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,1].set_xlabel("Weights:\n" +
                       "Convolved Image: " + str(round(second_model.coef_[0][0], 3)) + "\n" +
                       "GLCM Tiles:      " + str(round(second_model.coef_[0][1], 3)) + "\n" +
                       "Infrared Image:  " + str(round(second_model.coef_[0][2], 3)))

    ax[1,2].set_title("Class Weights:" + "\n" +
                             "0: " + str(round(models["Class Weights"][2][0],3)) + ", " +
                             "1: " + str(round(models["Class Weights"][2][1],3)))
    ax[1,2].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,2].imshow(ones, alpha = (pred_three*pred_three_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,2].set_xlabel("Weights:\n" +
                       "Convolved Image: " + str(round(third_model.coef_[0][0], 3)) + "\n" +
                       "GLCM Tiles:      " + str(round(third_model.coef_[0][1], 3)) + "\n" +
                       "Infrared Image:  " + str(round(third_model.coef_[0][2], 3)))

    ax[1,3].set_title("Class Weights:" + "\n" +
                             "0: " + str(round(models["Class Weights"][3][0],3)) + ", " +
                             "1: " + str(round(models["Class Weights"][3][1],3)))
    ax[1,3].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,3].imshow(ones, alpha = (pred_four*pred_four_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,3].set_xlabel("Weights:\n" +
                       "Convolved Image: " + str(round(fourth_model.coef_[0][0], 3)) + "\n" +
                       "GLCM Tiles:      " + str(round(fourth_model.coef_[0][1], 3)) + "\n" +
                       "Infrared Image:  " + str(round(fourth_model.coef_[0][2], 3)))

    ax[1,4].set_title("Class Weights:" + "\n" +
                             "0: " + str(round(models["Class Weights"][4][0],3)) + ", " +
                             "1: " + str(round(models["Class Weights"][4][1],3)))
    ax[1,4].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,4].imshow(ones, alpha = (pred_five*pred_five_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,4].set_xlabel("Weights:\n" +
                       "Convolved Image: " + str(round(fifth_model.coef_[0][0], 3)) + "\n" +
                       "GLCM Tiles:      " + str(round(fifth_model.coef_[0][1], 3)) + "\n" +
                       "Infrared Image:  " + str(round(fifth_model.coef_[0][2], 3)))

    ax[1,5].set_title("Balanced Model\nPredicted GLCM Tiles\n" + "Class Weights:\n" +
                       "0: " + str(round(models["Class Weights"][5][0],3)) + ", " +
                       "1: " + str(round(models["Class Weights"][5][1],3)))
    ax[1,5].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,5].imshow(ones, alpha = (pred_bal*pred_bal_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,5].set_xlabel("Weights:\n" +
                       "Convolved Image: " + str(round(balanced_model.coef_[0][0], 3)) + "\n" +
                       "GLCM Tiles:      " + str(round(balanced_model.coef_[0][1], 3)) + "\n" +
                       "Infrared Image:  " + str(round(balanced_model.coef_[0][2], 3)))

    for i in range(0,2):
        for j in range(0,6):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    plt.tight_layout()
    plt.show()
    filepath = r'/home/nmitchell/GLCM/initial-log-reg-images/'
    filepath += 'InitialLogReg__' + str(isamp) + ".png"
    fig.savefig(filepath)
    print(filepath)
    fig.close()


