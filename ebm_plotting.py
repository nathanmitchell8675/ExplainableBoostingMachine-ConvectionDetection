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

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
#####################################################

tile_size = 4

filepath = r'/home/nmitchell/GLCM/models/EBM_model'
with open(filepath, 'rb') as file:
    model = pkl.load(file)

ebm = model["Model"][0]

validation_data = xr.open_dataset('/home/nmitchell/GLCM/validation_data.nc')

ones = np.ones((int(256/tile_size), int(254/tile_size)))
red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["red", "black"])
gt_cm  = mpl.colors.LinearSegmentedColormap.from_list(" ", ["white", "red"])

#y_pred = ebm.predict(X_val)
#y_pred = [float(i) for i in y_pred]
#print(metrics.confusion_matrix(y_val, y_pred))

#505, 509

num1 = 1
num2 = 150

for isamp in range(num1,num2):
    #Pull the Ground Truth, the Expanded Ground Truth, the Convolved Image, the Original GLCM Tiles,
    #and the Infrared Image from the testing dataset. To change images, change the Sample number (isamp).
    val_gt     = validation_data.Ground_Truth.sel(Sample = isamp).values.flatten()
    val_egt    = validation_data.Expanded_Ground_Truth.sel(Sample = isamp).values.flatten()

    val_conv   = preprocessing.normalize(validation_data.Convolved_Image.sel(Sample = isamp).values.reshape(-1,1), axis = 0).flatten()
    val_glcm_a = preprocessing.normalize(validation_data.Above_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(-1,1), axis = 0).flatten()
    val_glcm_b = preprocessing.normalize(validation_data.Below_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(-1,1), axis = 0).flatten()
    val_ir     = preprocessing.normalize(validation_data.Infrared_Image.sel(Sample = isamp).values.reshape(-1,1), axis = 0).flatten()

    X_val  = np.transpose(pd.DataFrame((val_conv, val_glcm_a, val_glcm_b, val_ir), index = ["Convolved Data", "GLCM Above", "GLCM Below", "Infrared Data"]))
    y_val  = val_egt

    #Predicts the GLCM values for the reserved testing image
    pred_glcm = ebm.predict(X_val)
    pred_glcm = [float(i) for i in pred_glcm]
#    pred_one_probs   = first_model.predict_proba(X_val)[:,1]

    fig, ax = plt.subplots(4,6)

    convolved_image  = validation_data.Convolved_Image.sel(Sample = isamp).values.reshape(64,64)
    ground_truth     = val_gt.reshape(64,64)
    exp_ground_truth = val_egt.reshape(64,64)
    original_image   = validation_data.Original_Image.sel(Sample = isamp).values.reshape(256,256)
    infrared_image   = validation_data.Infrared_Image.sel(Sample = isamp).values.reshape(64,64)

    #          #              #           # cv image # cv image IMPORTANCE
    # og image # ground truth # pred glcm # exp gt   # gt IMPORTANCE
    #          #              #           # og glcm  # og glcm IMPORTANCE

#    print(len(ebm.term_scores_))
#    print(ebm.term_names_)
#    print(ebm.term_features_)

    ### FIRST COLUMN ###
    ax[0,0].set_visible(False)

    ax[1,0].set_title("Original Image")
    ax[1,0].imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))

    ax[2,0].set_visible(False)

#    ebm_explanation = ebm.explain_global()
#    ax[3,0].imshow(ebm_explanation.visualize(0))
    #ax[3,0].plot(ebm.term_scores_[0])
    ax[3,0].set_visible(False)

    ### SECOND COLUMN ###
    ax[0,1].set_visible(False)

    ax[1,1].set_title("Original Ground Truth")
    ax[1,1].imshow(ground_truth, cmap = gt_cm, origin = 'lower', extent = (0,64,0,64))

    ax[2,1].set_visible(False)

    ax[3,1].set_visible(False)

    ### THIRD COLUMN ###
    ax[0,2].set_visible(False)

    ax[1,2].set_title("Predicted Convection")
    ax[1,2].imshow(convolved_image, cmap = 'gray', origin = 'lower', extent = (0,64,0,64))
    ax[1,2].imshow(ones, alpha = (np.array(pred_glcm).reshape(64,64)*ebm.predict_proba(X_val)[:,1].reshape(64,64)), cmap = red_cm, origin = 'lower', extent = (0,64,0,64))

    ax[2,2].set_visible(False)

    ax[3,2].set_visible(False)

    ### FOURTH COLUMN ###
    ax[0,3].set_title("Convolved Image")
    ax[0,3].imshow(convolved_image, cmap = 'gray', origin = 'lower', extent = (0,64,0,64))

    ax[1,3].set_title("Infrared Image")
    ax[1,3].imshow(infrared_image, cmap = 'gray', origin = 'lower', extent = (0,64,0,64))

    ax[2,3].set_title("GLCM < 250K")
    ax[2,3].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[2,3].imshow(ones, alpha = validation_data.Below_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(64,64), cmap = red_cm, origin = 'lower', extent = (0,64,0,64))

    ax[3,3].set_title("GLCM > 250K")
    ax[3,3].imshow(convolved_image, cmap = 'gray', origin = 'lower')
    ax[3,3].imshow(ones, alpha = validation_data.Above_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(64,64), cmap = red_cm, origin = 'lower', extent = (0,64,0,64))

    ### FIFTH COLUMN ###
    ax[0,4].set_title("Convolved Image\nImportance")
    cv_importance = ax[0,4].imshow(ebm.eval_terms(X_val)[:,0].reshape(64,64), cmap = 'Spectral', origin = 'lower', extent = (0,64,0,64))
    plt.colorbar(cv_importance, ax = ax[0,4], location = 'left')

    ax[1,4].set_title("IR Image")
    glcm_a_importance = ax[1,4].imshow(ebm.eval_terms(X_val)[:,3].reshape(64,64), cmap = 'Spectral', origin = 'lower', extent = (0,64,0,64))
    plt.colorbar(glcm_a_importance, ax = ax[1,4], location = 'left')

    ax[2,4].set_title("GLCM <250k")
    glcm_importance = ax[2,4].imshow(ebm.eval_terms(X_val)[:,2].reshape(64,64), cmap = 'Spectral', origin = 'lower', extent = (0,64,0,64))
    plt.colorbar(glcm_importance, ax = ax[2,4], location = 'left')

    ax[3,4].set_title("GLCM >250K")
    infrared_importance = ax[3,4].imshow(ebm.eval_terms(X_val)[:,1].reshape(64,64), cmap = 'Spectral', origin = 'lower', extent = (0,64,0,64))
    plt.colorbar(glcm_importance, ax = ax[3,4], location = 'left')

    ### SIXTH COLUMN ###
    ax[0,5].plot(ebm.term_scores_[0])
    ax[0,5].set_aspect(100)
#    ax[0,5].set_ylabel("Score")

    ax[1,5].plot(ebm.term_scores_[3])
#    ax[1,5].set_aspect(100)

    ax[2,5].plot(ebm.term_scores_[2])
    ax[2,5].set_aspect(150)

    ax[3,5].plot(ebm.term_scores_[1])
    ax[3,5].set_aspect(150)

    for i in range(0,4):
        for j in range(0,5):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

#    print(ebm.term_importances(importance_type = "avg_weight"))
#    print(ebm.bagged_intercept_)

    fig.set_size_inches((8.5, 11), forward=False)
    plt.subplots_adjust(left = 0.012, bottom = 0.045, right = 0.979, top = 0.938, wspace = 0, hspace = 0.245)
    plt.show()

    filepath = r'/home/nmitchell/GLCM/EBM-no-interaction/'
    filepath += 'EBM_no_interaction_' + str(isamp) + ".png"
    fig.savefig(filepath)
    print(filepath)
    plt.close()

