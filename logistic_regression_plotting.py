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

tile_size = 4

filepath = r'/home/nmitchell/GLCM/models/log_reg_models'
with open(filepath, 'rb') as file:
    models = pkl.load(file)

first_model  = models["Models"][0]
second_model = models["Models"][1]
third_model  = models["Models"][2]
fourth_model = models["Models"][3]
fifth_model  = models["Models"][4]
sixth_model  = models["Models"][5]

validation_data = xr.open_dataset('/home/nmitchell/GLCM/validation_data.nc')

ones = np.ones((int(256/tile_size), int(254/tile_size)))
red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["red", "black"])
gt_cm  = mpl.colors.LinearSegmentedColormap.from_list(" ", ["white", "red"])

num1 = 9
num2 = 10

for isamp in range(num1,num2):
    #Pull the Ground Truth, the Expanded Ground Truth, the Convolved Image, the Original GLCM Tiles,
    #and the Infrared Image from the testing dataset. To change images, change the Sample number (isamp).
    val_gt    = validation_data.Ground_Truth.sel(Sample = isamp).values.flatten()
    val_egt   = validation_data.Expanded_Ground_Truth.sel(Sample = isamp).values.flatten()

    val_conv  = preprocessing.normalize(validation_data.Convolved_Image.sel(Sample = isamp).values.reshape(-1,1), axis = 0).flatten()
    val_glcm  = preprocessing.normalize(validation_data.Infrared_Mask_Applied_to_Min_Mask.sel(Sample = isamp).values.reshape(-1,1), axis = 0).flatten()
    val_ir    = preprocessing.normalize(validation_data.Infrared_Image.sel(Sample = isamp).values.reshape(-1,1), axis = 0).flatten()

    X_val  = np.transpose(pd.DataFrame((val_conv, val_glcm, val_ir), #, val_conv*val_glcm, val_ir*val_glcm),
                                        index = ["Convolved Data", "GLCM Data", "Infrared Data"])) #, "Brightness/GLCM Interaction", "Infrared/GLCM Interaction"]))

    y_val  = val_egt

    #Predicts the GLCM values for the reserved testing image
    pred_one         = first_model.predict(X_val)
    pred_one_probs   = first_model.predict_proba(X_val)[:,1]

    pred_two         = second_model.predict(X_val)
    pred_two_probs   = second_model.predict_proba(X_val)[:,1]

    pred_three       = third_model.predict(X_val)
    pred_three_probs = third_model.predict_proba(X_val)[:,1]

    pred_four        = fourth_model.predict(X_val)
    pred_four_probs  = fourth_model.predict_proba(X_val)[:,1]

    pred_five        = fifth_model.predict(X_val)
    pred_five_probs  = fifth_model.predict_proba(X_val)[:,1]

    pred_six         = sixth_model.predict(X_val)
    pred_six_probs   = sixth_model.predict_proba(X_val)[:,1]

    #Function to create the text for the title of the second row of plots
    def create_title(model_num):
        title = ('Class Weights:\n' +
                       "0: " + str(round(models["Class Weights"][model_num][0],3)) + ", " +
                       "1: " + str(round(models["Class Weights"][model_num][1],3)))
        return title

    #Function to get the number of false positives and false negatives for each image
    def fp_fn(preds):
        false_pos = 0
        false_neg = 0

        for i in range(len(preds)):
            if(preds[i] == 1):
                if(ground_truth.flatten()[i] == 0):
                    false_pos = false_pos + 1
            if(ground_truth.flatten()[i] == 1):
                if(preds[i] == 0):
                    false_neg = false_neg + 1

        return false_pos, false_neg

    #Function to create the x label for each image in the second row
    def create_xlabel(pred_model, preds):
        false_pos, false_neg = fp_fn(preds)
        label = ("Weights:\n" +
                "Convolved Image: " + str(round(pred_model.coef_[0][0], 3)) + "\n" +
                "GLCM Tiles:      " + str(round(pred_model.coef_[0][1], 3)) + "\n" +
                "Infrared Image:  " + str(round(pred_model.coef_[0][2], 3)) + "\n\n" +
                "False Positives: " + str(false_pos) + "\n" +
                "False Negatives: " + str(false_neg))
        return label

    #Initializes the Plots
    fig, ax = plt.subplots(2,6)

    #Creates the convolved image, ground truth, and expanded ground truth
    convolved_image  = validation_data.Convolved_Image.sel(Sample = isamp).values.reshape(64,64)
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
    ax[0,4].imshow(ones, alpha = validation_data.Infrared_Mask_Applied_to_Min_Mask.sel(Sample = isamp).values.reshape(64,64), cmap = red_cm, origin = 'lower')

    ax[0,5].set_visible(False)

    ### SECOND ROW ###
    ax[1,0].set_title("Unbalanced Model\nPredicted GLCM Tiles\n" + create_title(0))
    ax[1,0].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,0].imshow(ones, alpha = (pred_one*pred_one_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,0].set_xlabel(create_xlabel(first_model, pred_one))

    ax[1,1].set_title(create_title(1))
    ax[1,1].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,1].imshow(ones, alpha = (pred_two*pred_two_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,1].set_xlabel(create_xlabel(second_model, pred_two))

    ax[1,2].set_title(create_title(2))
    ax[1,2].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,2].imshow(ones, alpha = (pred_three*pred_three_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,2].set_xlabel(create_xlabel(third_model, pred_three))

    ax[1,3].set_title(create_title(3))
    ax[1,3].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,3].imshow(ones, alpha = (pred_four*pred_four_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,3].set_xlabel(create_xlabel(fourth_model, pred_four))

    ax[1,4].set_title(create_title(4))
    ax[1,4].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,4].imshow(ones, alpha = (pred_five*pred_five_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,4].set_xlabel(create_xlabel(fifth_model, pred_five))

    ax[1,5].set_title("Balanced Model\nPredicted GLCM Tiles\n" + create_title(5))
    ax[1,5].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[1,5].imshow(ones, alpha = (pred_six*pred_six_probs).reshape(64,64), cmap = red_cm, origin = 'lower')
    ax[1,5].set_xlabel(create_xlabel(sixth_model, pred_six))

    for i in range(0,2):
        for j in range(0,6):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

#    fig = plt.gcf()
#    fig.set_size_inches((8.5, 11), forward=False)
#    plt.tight_layout()
    plt.show()
#    filepath = r'/home/nmitchell/GLCM/initial-log-reg-images/'
#    filepath += 'InitialLogReg__' + str(isamp) + ".png"
#    fig.savefig(filepath)
#    print(filepath)
#    plt.close()


