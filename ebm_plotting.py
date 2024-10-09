import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import axes
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xarray as xr
import pickle as pkl
import xarray as xr
from numpy import random

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from matplotlib import ticker

###
from skimage.restoration import inpaint
from scipy.ndimage import laplace
from scipy.ndimage import sobel
import copy
from skimage.feature import graycomatrix, graycoprops
import cv2 as cv

import math
import warnings
warnings.filterwarnings('ignore')
#####################################################

tile_size = 4

#Load in the data
validation_data = xr.open_dataset('/home/nmitchell/GLCM/training_data_fixStripes_all.nc')

#Filepath for the EBM
filepath = r'/home/nmitchell/GLCM/models/EBM_model_ADASYN'

### Model Loading ###
with open(filepath, 'rb') as file:
    model = pkl.load(file)
ebm = model["Model"][0]

#ebm.scale(7, 1.25)

### Feature Renaming ###
names = ebm.term_names_
names = '  '.join(names)

feature_names = ['Brightness', 'Warm GLCM', 'Cool GLCM', 'Infrared Image']
for i in range(len(feature_names)):
    names = names.replace('feature_000' + str(i), feature_names[i])

names = names.split('  ')

#Sort the scores in descending order and sort the names along with them
all_scores, all_names = zip(*sorted(zip(ebm.explain_global().data()['scores'], names)))

### Intercept Alteration ###
global intercept_addition #2.25
intercept_addition = 2.25
ebm.intercept_ = ebm.intercept_ + intercept_addition


### Shape Function Placeholders ###
global edited_shape_fxs
edited_shape_fxs = np.array([])

global edit_range
edit_range = np.array([])

def alter_model():
    global edited_shape_fxs
    edited_shape_fxs = np.array([])

    global edit_range
    edit_range = np.array([])

    ### Shape Function Alteration ###
    ### f(x): Convolved Image (0) ###
    xvals = np.array(ebm.explain_global().data(0)['names'][1:1023])
    #2.6
    #ebm.explain_global().data(0)['scores'][:] = np.where((xvals <= 29), xvals*0.05 - 2.6, ebm.explain_global().data(0)['scores'])

    ebm.explain_global().data(0)['scores'][:] = np.where((xvals <= 29), -1.225, ebm.explain_global().data(0)['scores'])

    #For setting the bounds to 0 later on, need the name and which positions to switch
    edited_shape_fxs = np.append(edited_shape_fxs, ["brightness"])
    edit_range = np.hstack((edit_range, np.array([0, np.sum(xvals <= 29)])))


#    ebm.explain_global().data(1)['scores'][0] = 0.25

    ### f(x): Cool GLCM (2) ###
    xvals = np.array(ebm.explain_global().data(2)['names'][1:1023])
    yvals = np.array(ebm.explain_global().data(2)['scores'])

    #4, 1.5, 1, 2
    ebm.explain_global().data(2)['scores'][:] = (xvals*4 + yvals/1.5 - 1)*2
    ebm.explain_global().data(2)['scores'][986:1022] = yvals[986:1022]


    #For setting the bounds to 0 later on, need the name and which positions to switch
    edited_shape_fxs = np.append(edited_shape_fxs, ["cool_glcm"])
    edit_range = np.vstack((edit_range, np.array([0,986])))

    ### f(x): Interactions ###
    ## Brightness/Cool GLCM ##
    x = np.array(ebm.explain_global().data(5)['left_names'])
    y = np.array(ebm.explain_global().data(5)['right_names'])
    z = np.array((ebm.explain_global().data(5)['scores'].T))

    (ebm.explain_global().data(5)['scores'].T)[:] = np.where(z >= 2, z - 2, z)
    #z = (ebm.explain_global().data(5)['scores'].T)[:]

    (ebm.explain_global().data(5)['scores'].T)[0:11,0:31] = (ebm.explain_global().data(5)['scores'].T)[29:30,0:11].T
    (ebm.explain_global().data(5)['scores'].T)[0:29,0:11] = (ebm.explain_global().data(5)['scores'].T)[29:30, 0:11]

    ## Brightness/InfraredImage ##
    x = np.array(ebm.explain_global().data(6)['left_names'])
    y = np.array(ebm.explain_global().data(6)['right_names'])
    z = np.array(ebm.explain_global().data(6)['scores'].T)

    z[0:30,0:10] = z[0:30,0:10] - 2

    (ebm.explain_global().data(6)['scores'].T)[:] = z #np.where((z_2 > 0) & (z_2 <= 0.25), z*2, z)

    ## Infrared Image/Cool GLCM ##
    x = np.array(ebm.explain_global().data(7)['left_names'])
    y = np.array(ebm.explain_global().data(7)['right_names'])

    x_stop = np.sum(x <= 0.025)
    y_stop = np.sum(y <= 225)
    (ebm.explain_global().data(7)['scores'].T)[0:y_stop, 0:x_stop] = (((ebm.explain_global().data(7)['scores'].T)[0:y_stop, 0:x_stop])/1) - 1

alter_model()

###########################

def get_statistics():
    brightness = validation_data.Convolved_Image.values.reshape(-1,1).flatten()
    infrared   = validation_data.Infrared_Image.values.reshape(-1,1).flatten()
    warm_glcm  = validation_data.Above_IR_Mask_Applied_to_OG_GLCM.values.reshape(-1,1).flatten()
    cool_glcm  = validation_data.Below_IR_Mask_Applied_to_OG_GLCM.values.reshape(-1,1).flatten()

    X_val = np.transpose(np.array([brightness, warm_glcm, cool_glcm, infrared]))
    y_val = validation_data.Masked_Truth.values.flatten()

    predictions       = ebm.predict(X_val)
    pred_convection   = np.array([float(i) for i in predictions]).reshape(-1,1)

    y_pred = []

    def remove_bits():
        for k in range(int(len(pred_convection)/4096)):
            image   = pred_convection[k*(4096):k*(4096)+4096].reshape(64,64)
            image_2 = pred_convection[k*(4096):k*(4096)+4096].reshape(64,64)
            N = len(image)
            for i in range(N):
                for j in range(N):
                    total = int(image[(i-1)%N,(j-1)%N] + image[(i-1)%N,j] + image[(i+1)%N,(j+1)%N] +
                    image[i,(j-1)%N]                                      + image[i,(j+1)%N] +
                    image[(i+1)%N,(j-1)%N]             + image[(i+1)%N,j] + image[(i-1)%N,(j+1)%N])

                    if((total <= 1) & (image[i,j] == 1.)):
                        image_2[i,j] = 0

            y_pred[k*(4096):k*(4096)+4096] = image_2.flatten()


    #remove_bits()
    y_pred = np.array([float(i) for i in predictions]).reshape(-1,1)

    tn, fp, fn, tp = metrics.confusion_matrix(y_val, y_pred).ravel()

    print("Edited Model:")
    print("True Positives:  ", tp)
    print("True Negatives:  ", tn)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)

#get_statistics()

#Create useful colormaps to be used while plotting
ones = np.ones((int(256/tile_size), int(254/tile_size)))
red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["crimson", "black"])
blu_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["mediumblue", "black"])
gt_cm  = mpl.colors.LinearSegmentedColormap.from_list(" ", ["white", "red"])

o_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["orange", "black"])

#examples = random.randint(6033, size = (5))
#print(examples)

#Favorite Examples:
#examples = [379,379,417,588,773,781]

# -=+ TESTING DATASET +=- #
#WORST Examples
#examples = [26,57,88,119,150,181,207,212,238,243,269,274,305,332,336,363,367,394,398,425,429,430,456,460,461,487,491,492,518]
#examples = [522,523,546,553,554,577,584,585,615,616,646,672,677,701,703,708,732,734,735,739,757,765,766,770,794,825,828,833]

#examples = [758,300,243]

#examples = [350,540,610,645,1429,1470,1511,1551,1591,1674,1716,1758,1800,1959,2001,2043,2085,2092,2134,2176,2218,2239,2281,2323,2324,2360,2365,2380,2402,2403,2404,3017,3044,3071,4080,4081,4099,4102,4466,4795,4826,4857,4887,4917,5041]

examples = [1150,731,3337]

#for isamp in range(num1,num2):
for isamp in examples:
    #Pulls all of the variables needed to create X. X is needed to make predictions.
    #Data Pulled: Convoled Image, GLCM Above Threshold, GLCM Below Threshold, Infrared Image
    brightness = validation_data.Convolved_Image.sel(Sample = isamp).values.reshape(-1,1).flatten()
    infrared   = validation_data.Infrared_Image.sel(Sample = isamp).values.reshape(-1,1).flatten()
    warm_glcm  = validation_data.Above_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(-1,1).flatten()
    cool_glcm  = validation_data.Below_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(-1,1).flatten()

    X_val = np.transpose(np.array([brightness, warm_glcm, cool_glcm, infrared]))
    #y_val = validation_data.Masked_Truth.values.flatten()

    #Pull Remaining Images & Reshape
    full_mrms      = validation_data.Ground_Truth.sel(Sample = isamp).values.reshape(int(256/tile_size), -1)
    masked_mrms    = validation_data.Masked_Truth.sel(Sample = isamp).values.reshape(int(256/tile_size), -1).astype(np.uint8)
    brightness     = brightness.reshape(64,64)
    original_image = validation_data.Original_Image.sel(Sample = isamp).values.reshape(256,256)
    infrared_image = validation_data.Infrared_Image.sel(Sample = isamp).values.reshape(64,64)
    #big_mrms       = validation_data.Full_Sized_MRMS.sel(Sample = isamp).values.reshape(256,256)

#    for i in range(len(original_image.flatten())):
#        print(str(original_image.flatten()[i]) + ',', sep = '', end = '')

    #Pull the Warm (> 250K) & Cool (< 250K) GLCM Tiles & Reshape
    warm_glcm = validation_data.Above_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(64,64)
    cool_glcm = validation_data.Below_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(64,64)
    #########################################

    def big_shape_function(a):
        fig, ax = plt.subplots(1,4)
        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        gs = ax[0].get_gridspec()

        ax[0].imshow(brightness, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))

        l = 11
        u = 29

        shade_red = np.where((brightness >= l) & (brightness <= u), 1, np.nan)

        ax[1].imshow(brightness, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        ax[1].imshow(shade_red, cmap = red_cm, origin = 'lower', extent = (0,256,0,256), alpha = 0.5)

        shape_fx = fig.add_subplot(gs[2:4])
        shape_fx.grid(True, linestyle = ':')
        shape_fx.set_axisbelow(True)

        #Set the x- and y-values for plotting purposes (currently removing the first entry to ensure arrays are of the same length, for plotting)
        xvals = np.array(ebm.explain_global().data(a)['names'][1:len(ebm.explain_global().data(a)['names'])])
        yvals = np.array(ebm.explain_global().data(a)['scores'])

        lower_bounds = np.array(ebm.explain_global().data(a)['lower_bounds'])
        upper_bounds = np.array(ebm.explain_global().data(a)['upper_bounds'])

        xvals_g = np.where((xvals <= l) | (xvals >= u), xvals, np.nan)
        xvals_r = np.where((xvals > l) & (xvals < u), xvals, np.nan)

        lower_g = np.where((xvals <= l) | (xvals >= u), lower_bounds, np.nan)
        lower_r = np.where((xvals > l) & (xvals < u), lower_bounds, np.nan)

        upper_g = np.where((xvals <= l) | (xvals >= u), upper_bounds, np.nan)
        upper_r = np.where((xvals > l) & (xvals < u), upper_bounds, np.nan)

        #Plot a dotted line at y = 0 for easy comparison across the full plot
        shape_fx.plot(xvals, np.zeros(len(xvals)), ':', color = 'black')
        shape_fx.step(xvals_g, yvals, color = 'dimgray')
        shape_fx.step(xvals_r, yvals, color = 'crimson')

        shape_fx.fill_between(xvals, lower_g, upper_g, color = 'dimgray', alpha = 0.25)
        shape_fx.fill_between(xvals, lower_r, upper_r, color = 'crimson', alpha = 0.25)

        fig.set_size_inches((8.5, 11), forward=False)
        plt.subplots_adjust(left = 0.012, bottom = 0.045, right = 0.971, top = 0.938, wspace = 0.2, hspace = 0.245)
        plt.show()

#    big_shape_function(0)
#    continue

    def int_shape(b):
        fig, ax = plt.subplots(1,2)

        max_cm = max(ebm.eval_terms(X_val).flatten())
        min_cm = min(ebm.eval_terms(X_val).flatten())
        plot_max = max(abs(max_cm), abs(min_cm))

        IM = np.array(ebm.eval_terms(X_val)[:,b])

        #print(max(IM) - 1.25)

        importance = ax[0].imshow(IM.reshape(64,64), cmap = 'seismic_r', origin = 'lower', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        plt.colorbar(importance, ax = ax[0], location = 'bottom', fraction=0.046, pad=0.04)

        #ax[0].imshow(np.where((IM >= max(IM) - 1.25), 1, np.nan).reshape(64,64), cmap = o_cm, origin = 'lower', extent = (0,64,0,64))

        x = np.array(ebm.explain_global().data(b)['left_names'])
        y = np.array(ebm.explain_global().data(b)['right_names'])
        z = np.array(ebm.explain_global().data(b)['scores'].T)

        x_start = np.sum(x<=20)
        x_stop = len(x)-np.sum(x>=85)

        y_start = np.sum(y <= 220)
        y_stop  = len(y)-np.sum(y>=250)

        area = z[y_start:y_stop, x_start:x_stop]
        area = np.where(area <= 0, area*-3, area)

        z[y_start:y_stop, x_start:x_stop] = area

        interaction = ax[1].pcolormesh(x,y,z, cmap = 'seismic_r', vmin = -plot_max, vmax = plot_max)
        plt.colorbar(interaction, ax = ax[1],location = 'right', fraction = 0.046, pad = 0.04)
        plt.show()

#    int_shape(5)
#    continue

    def slim_plotting():
        fig, ax = plt.subplots(1,3)
        for i in range(3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        #Original Image
        ax[0].imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        ax[0].imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        ax[0].set_title("Channel 2: Visible Imagery")

        #Multi-Radar Multi-Sensor
        ax[1].imshow(original_image, alpha = 0.5,   cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        ax[1].imshow(ones, alpha = full_mrms*0.25, cmap = red_cm, origin = 'lower', extent = (0,256,0,256)) #Full Ground Truth (low opacity)
        ax[1].imshow(ones, alpha = masked_mrms,       cmap = red_cm, origin = 'lower', extent = (0,256,0,256)) #IR Masked Ground Truth (full opacity)
        ax[1].set_title("Multi-Radar Multi-Sensor (MRMS)")

        #Predicted Convection
        predictions     = ebm.predict(X_val)
        pred_convection = np.array([float(i) for i in predictions]).reshape(int(256/tile_size), int(256/tile_size))

        ax[2].imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        ax[2].imshow(ones, alpha = (np.array(pred_convection)*ebm.predict_proba(X_val)[:,1].reshape(64,64)), cmap = red_cm, origin = 'lower', extent = (0,256,0,256))
        ax[2].set_title("Predicted Convection")


        print(isamp)
#        print(np.sum(masked_mrms))
#        print(np.sum(pred_convection))
#        print("Sample: ", isamp)
#        print("Overprediction: ", np.sum(pred_convection) - np.sum(masked_mrms))

        plt.show()

    slim_plotting()
    continue

    #########################################

    #Create the plotting interface
    fig, ax = plt.subplots(4,10)
    gs = ax[0,0].get_gridspec()

    ### USEFUL FUNCTIONS ###

    #Function to remove ticks & axes from small images
    def remove_ax_labels(b):
        for i in range(4):
            for a in ax[0:4,i]:
                a.remove()
        for i in range(0,4):
            for j in range(0,b):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])

    #Function to remove ticks from large images
    def remove_ticks(ax, title):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, y = 0.99)

    #Creates the "Large Images," i.e. the Original Image, Ground Truth, and Predicted Convection
    def big_images(original_features, interactions):

        ### Original Image ###
        og_im_ax = fig.add_subplot(gs[0:2, 0:2])
        og_im_ax.imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        og_im_ax.yaxis.set_label_position("right")
        intercept  = "Intercept: " + str(np.round(ebm.intercept_[0], 3)) + "\n\n\n\n\nSample Number: " + str(isamp)
        intercept2 = "\nOriginal: " + str(np.round(ebm.intercept_[0], 3) - intercept_addition)
        og_im_ax.set_ylabel(intercept, rotation = 0, fontsize = 15)
        og_im_ax.set_xlabel(intercept2, fontsize = 10)
        og_im_ax.xaxis.set_label_coords(1.65, 0.725)
        og_im_ax.yaxis.set_label_coords(1.65, 0.75)
        remove_ticks(og_im_ax, "Channel 2: Visible Imagery")
        ###----------------###

        ### Ground Truth ###
        ones2 = np.ones((256,256))
        mrms_ax = fig.add_subplot(gs[2:4, 0:2])
        mrms_ax.imshow(original_image, alpha = 0.5,  cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        mrms_ax.imshow(ones, alpha = full_mrms*0.25, cmap = red_cm, origin = 'lower', extent = (0,256,0,256)) #Full Ground Truth (low opacity)
        mrms_ax.imshow(ones, alpha = masked_mrms,    cmap = red_cm, origin = 'lower', extent = (0,256,0,256)) #IR Masked Ground Truth (full opacity)
        remove_ticks(mrms_ax, "Multi-Radar Multi-Sensor (MRMS)")
        ###--------------###

        ### Predicted Convection ###
        predictions       = ebm.predict(X_val)

#        print(np.sum(predictions))

        pred_convection   = np.array([float(i) for i in predictions]).reshape(int(256/tile_size), int(256/tile_size))
        pred_convection_2 = np.array([float(i) for i in predictions]).reshape(int(256/tile_size), int(256/tile_size))

        #OPTIONAL: only keep predictions above a specific "certainty" threshold
#        certainty = 0.80
#        pred_convection    = np.array(np.where(np.array(ebm.predict_proba(X_val)[:,1] >= certainty).reshape(64,64), pred_convection_2, 0.0)).reshape(int(256/tile_size), int(256/tile_size))
#        pred_convection_2  = np.array(np.where(np.array(ebm.predict_proba(X_val)[:,1] >= certainty).reshape(64,64), pred_convection_2, 0.0)).reshape(int(256/tile_size), int(256/tile_size))

        N = len(pred_convection)

        def remove_bits():
            for i in range(N):
                for j in range(N):
                    total = int(pred_convection[(i-1)%N,(j-1)%N] + pred_convection[(i-1)%N,j] + pred_convection[(i+1)%N,(j+1)%N] +
                            pred_convection[i,(j-1)%N]                                        + pred_convection[i,(j+1)%N] +
                            pred_convection[(i+1)%N,(j-1)%N]     + pred_convection[(i+1)%N,j] + pred_convection[(i-1)%N,(j+1)%N])

                    if((total <= 1) & (pred_convection[i,j] == 1.)):
                        pred_convection_2[i,j] = 0

#        remove_bits()

        pred_convection_ax = fig.add_subplot(gs[2:4, 2:4])
        pred_convection_ax.imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        pred_convection_ax.imshow(ones, alpha = (np.array(pred_convection_2)*ebm.predict_proba(X_val)[:,1].reshape(64,64)), cmap = red_cm, origin = 'lower', extent = (0,256,0,256))
        remove_ticks(pred_convection_ax, "Predicted Convection")
        ### ---------------------###


    ### Feature Imporance ###
    max_cm = max(ebm.eval_terms(X_val).flatten())
    min_cm = min(ebm.eval_terms(X_val).flatten())
    plot_max = max(abs(max_cm), abs(min_cm))

    def importance(a,b,c, title):
        ax[a,b].set_title(title)
        importance = ax[a,b].imshow(ebm.eval_terms(X_val)[:,c].reshape(64,64), cmap = 'seismic_r', origin = 'lower', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        plt.colorbar(importance, ax = ax[a,b], location = 'bottom', fraction=0.046, pad=0.04)
    ###-------------------###

    ### Shape Functions ###
    def shape_function(a, name, first_plot, last_plot):
        #Sets the data, based on the name passed
        data = eval(name).flatten()

        #Creates the plotting interface
        shape_fx = fig.add_subplot(gs[a, 6:8])
        shape_fx.grid(True, linestyle = ':')
        shape_fx.set_axisbelow(True)

        #Set the x- and y-values for plotting purposes (currently removing the first entry to ensure arrays are of the same length, for plotting)
        xvals = ebm.explain_global().data(a)['names'][1:len(ebm.explain_global().data(a)['names'])]
        yvals = ebm.explain_global().data(a)['scores']

        #Plot a dotted line at y = 0 for easy comparison across the full plot
        shape_fx.plot(xvals, np.zeros(len(xvals)), ':', color = 'black')

        #Find the min and max of both the shape function and the data
        min_shape_fx = np.min(xvals)
        max_shape_fx = np.max(xvals)

        min_data = np.min(data)
        max_data = np.max(data)

        #Empty array used in place of some data
        empty_array = np.full(len(xvals), np.nan)

        #Function to plot the shape functions themselves
        def color_shape_fx(x_convection, y_convection, x_non_convec, y_non_convec, x_no_data_, y_no_data_, a, b):
            if a:
                shape_fx.step(x_convection, y_convection, color = 'mediumblue', label = 'Convective')
            if not b:
                shape_fx.step(x_non_convec, y_non_convec, color = 'crimson', label = 'Non-Convective')
            shape_fx.step(x_no_data_, y_no_data_, color = 'dimgray', label = 'No Data')

        #Function to plot the error bars of the shape functions
        def color_error_bars(x_convec, lower_convec, upper_convec, x_no_convec, lower_no_convec, upper_no_convec, x_no_data, lower_no_data, upper_no_data, a, b):
            if a:
                shape_fx.fill_between(x_convec, lower_convec, upper_convec, alpha = 0.25, color = 'mediumblue')
            if not b:
                shape_fx.fill_between(x_no_convec, lower_no_convec, upper_no_convec, alpha = 0.25, color = 'crimson')
            shape_fx.fill_between(x_no_data, lower_no_data, upper_no_data, alpha = 0.25, color = 'dimgray')

        #Because the x-values aren't uniform & different lines are plotted (for colors), some lines must be extended
        def extend_values(data, vals):
            #Find the positions of the NaN values, save the first and last
            non_nan_indices = ~np.isnan(data)
            if(np.sum(non_nan_indices) != 0):
                lower = np.where(non_nan_indices)[0][0]
                upper = np.where(non_nan_indices)[0][-1]

                #Extend the values as long as you aren't at the end points
                if (lower != 0):
                    data[lower - 1] = vals[lower - 1]
                if ((upper != len(vals) - 2) & (upper != len(vals) - 1)):
                    data[upper + 1] = vals[upper + 1]

            return data

        #Determine the Upper and Lower Bounds -- must accommodate for places where the shape function has been edited
        if name in edited_shape_fxs:
            position = np.where(edited_shape_fxs == name)[0][0]
            bound_range = edit_range[position]
            arr = np.full(int(bound_range[1] - bound_range[0]), np.nan)

            #Where the shape function was edited, fill the upper/lower bounds with NaNs
            lower_bounds = np.array(ebm.explain_global().data(a)['lower_bounds'])
            lower_bounds[int(bound_range[0]):int(bound_range[1])] = arr

            upper_bounds = np.array(ebm.explain_global().data(a)['upper_bounds'])
            upper_bounds[int(bound_range[0]):int(bound_range[1])] = arr
        else:
            #If the shape functions weren't edited, the upper/lower bounds stay the same
            lower_bounds = ebm.explain_global().data(a)['lower_bounds']
            upper_bounds = ebm.explain_global().data(a)['upper_bounds']

        #Check to make sure the feature vector isn't empty (may be when IR is always > 250K)
        if(np.sum(data) != 0.0):
            #Range of x-values outside of the scope of the local feature
            no_data_x = extend_values(np.where((xvals <= min_data) | (xvals >= max_data), xvals, np.nan), xvals) #, which)
            #Range of error-bar values outside the score of the local feature
            no_data_lower = extend_values(np.where((xvals <= min_data) | (xvals >= max_data), lower_bounds, np.nan), lower_bounds)
            no_data_upper = extend_values(np.where((xvals <= min_data) | (xvals >= max_data), upper_bounds, np.nan), upper_bounds)

            #Next, check to see if there is any convection present within the local data
            if ((np.sum(data*masked_mrms.flatten()) == 0.0)): #No Convection Present (Locally)
                convection = False #Variable used to tell the plotting mechanism to not plot any convection
                #If there is no local convection, the min and max should be the min and max of the dataset
                min_ = max_data
                max_ = max_data
                #If there is no convection, the arrays used to plot convection should be empty
                convection_x_vals = np.full(len(xvals), np.nan)
                convection_lower  = np.full(len(xvals), np.nan)
                convection_upper  = np.full(len(xvals), np.nan)
            else: #Convection Present (Locally)
                convection = True #Variable used to tell the plotting mechanism to plot convection
                #If there is local convection, the min and max should be the min and max values for which convection has been observed
                min_ = np.nanmin(np.where(data*(masked_mrms.flatten()==1.0) == 0, np.nan, data*(masked_mrms.flatten()==1.0)))
                max_ = np.nanmax(np.where(data*(masked_mrms.flatten()==1.0) == 0, np.nan, data*(masked_mrms.flatten()==1.0)))

                #If the min and the max are equal, there is exactly one convective pixel
                if(min_ != max_):
                    #If there is local convection, the arrays used to plot the x-values and error bars should reflect the range of values
                    convection_x_vals = extend_values(np.where((xvals > min_) & (xvals < max_), xvals, np.nan), xvals)
                    convection_lower  = extend_values(np.where((xvals > min_) & (xvals < max_), lower_bounds, np.nan), lower_bounds)
                    convection_upper  = extend_values(np.where((xvals > min_) & (xvals < max_), upper_bounds, np.nan), upper_bounds)
                else:
                    #Create upper and lower bounds (should be the same number) to locate where the convective pixel is located within the shape function
                    upper_bound = len(xvals) - np.sum(xvals >= min_)
                    lower_bound = np.sum(xvals <= min_)
                    #Manually extend the range of x-values to be plotted from one to three (one on each side of the closest x-value corresponding to the convective pixel)
                    range_ = xvals[lower_bound - 2:upper_bound + 3]

                    min_ = min(range_)
                    max_ = max(range_)

                    #Flesh out the arrays used for plotting (adding nans) based on the range created above
                    convection_x_vals = np.where((xvals >= min(range_)) & (xvals <= max(range_)), xvals, np.nan)
                    convection_lower  = extend_values(np.where((xvals >= min(range_)) & (xvals <= max(range_)), lower_bounds, np.nan), lower_bounds)
                    convection_upper  = extend_values(np.where((xvals >= min(range_)) & (xvals <= max(range_)), upper_bounds, np.nan), upper_bounds)

            #If there is no local convection, this will span all values within the local image
            #If there is local convection, this will span any region in which there is no local convection
            #Changes based on what the values of max_ and min_ were set to previously
            non_convective_x  = np.where(((xvals < min_) & (xvals > min_data)) | ((xvals > max_) & (xvals < max_data)), xvals, np.nan)
            non_convec_lower  = extend_values(np.where(((xvals <= min_) & (xvals >= min_data)) | ((xvals >= max_) & (xvals <= max_data)), lower_bounds, np.nan), lower_bounds)
            non_convec_upper  = extend_values(np.where(((xvals <= min_) & (xvals >= min_data)) | ((xvals >= max_) & (xvals <= max_data)), upper_bounds, np.nan), upper_bounds)

            no_convection = False
            #Make sure there are areas of no convection
            if(len(non_convective_x) - np.nansum(non_convective_x) == len(non_convective_x)):
                no_convection = True

            #Calls the functions that are responsible for plotting the shape functions and their error bars
            color_shape_fx(convection_x_vals, yvals, non_convective_x, yvals, no_data_x, yvals, convection, no_convection)
            color_error_bars(convection_x_vals, convection_lower, convection_upper, non_convective_x, non_convec_lower, non_convec_upper, no_data_x, no_data_lower, no_data_upper, convection, no_convection)
        #If the feature is empty, plot the shape function as all gray (no data available)
        else:
            shape_fx.plot(xvals, yvals, color = 'dimgray')
            shape_fx.fill_between(xvals, ebm.explain_global().data(a)['lower_bounds'], ebm.explain_global().data(a)['upper_bounds'], alpha = 0.5, color = 'dimgray')

        #Checks to see if the first/last shape function is being plotted, labels accordingly
        if(first_plot):
            shape_fx.set_title("Shape Function:", y = 1.15)
            shape_fx.legend(bbox_to_anchor=(1.03, 1.175),fontsize=7,ncol = 3, columnspacing=0.8)
        if(last_plot):
            shape_fx.set_xlabel("Shape Function: Global\nColors: Local", fontsize=7.5)

        #Arranges the y-axis ticks of the plot
        shape_fx.yaxis.set_major_locator(ticker.FixedLocator(np.round(np.arange(-2, 6, 1.0).astype(int), 0)))
        shape_fx.yaxis.set_major_formatter(ticker.FixedFormatter(np.round(np.arange(-2, 6, 1.0).astype(int), 0)))

        return shape_fx, min(xvals), max(xvals)

    def plot_limits(ax, lower_x, upper_x, step_x, lab_type, round_by):
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.round(np.arange(lower_x, upper_x, step_x).astype(lab_type), round_by)))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(np.round(np.arange(lower_x, upper_x, step_x).astype(lab_type), round_by)))
        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[1::2]]
    ###-----------------###

    ### Feature Histograms ###
    def feat_hist(a, name, first_plot, last_plot):
        data = eval(name).flatten()
        hist = fig.add_subplot(gs[a,8:10])
        hist.grid(True, linestyle = ':')
        hist.set_axisbelow(True)
        bins = ebm.explain_global().data(a)['density']['names']

        if "cool" in name:
            n_mrms = masked_mrms.flatten()[infrared_image.flatten() <= 250]
            data = data.flatten()[infrared_image.flatten() <= 250]
        elif "warm" in name:
            n_mrms = masked_mrms.flatten()[infrared_image.flatten() > 250]
            data = data.flatten()[infrared_image.flatten() > 250]
        else:
            n_mrms = masked_mrms.flatten()

        hist.hist([np.where(n_mrms==0.0,data,np.nan), np.where(n_mrms==1.0,data,np.nan)],
                  bins = bins, histtype='bar',
                  stacked=True, label=['Non-Convective','Convective'], color=['crimson','mediumblue'])

        hist.tick_params(bottom=True, top=False, left=False, right=True)
        hist.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True)

        if(first_plot):
            hist.set_title("Feature Distribution:", y = 1.15)
            hist.legend(bbox_to_anchor=(0.925, 1.175),fontsize=7,ncol = 2)
        if(last_plot):
            hist.set_xlabel("Histogram: Local\nColors: Local", fontsize=7.5)

        return hist, min(bins), max(bins)

    ### ORIGINAL FEATURES ###

    #Convolved Image
    ax[0,4].set_ylabel(names[0], labelpad = 0.1)
    ax[0,4].set_title("Feature:")
    ax[0,4].imshow(brightness, cmap = 'gray', origin = 'lower', extent = (0,64,0,64))

    #Warm GLCM
    ax[1,4].set_ylabel(names[1], labelpad = 0.1)
    ax[1,4].imshow(ones, alpha = warm_glcm, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    #Cool GLCM
    ax[2,4].set_ylabel(names[2], labelpad = 0.1)
    ax[2,4].imshow(ones, alpha = cool_glcm, cmap = blu_cm, origin = 'lower', extent = (0,256,0,256))

    #Infrared Image
    ax[3,4].set_ylabel(names[3], labelpad = 0.1)
    ax[3,4].imshow(infrared_image, cmap = 'gray', origin = 'lower', extent = (0,64,0,64))


    ### FUNCTION CALLS ###
    big_images(True, False)

    #Plots the shape functions
    axis, min_x, max_x = shape_function(0, names[0].lower().replace(' ', '_'), True,False)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/10)*10)/10, int, 0)

    axis, min_x, max_x = shape_function(1, names[1].lower().replace(' ', '_'), False,False)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*1) + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

    axis, min_x, max_x = shape_function(2, names[2].lower().replace(' ', '_'), False,False)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*1) + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

    axis, min_x, max_x = shape_function(3, names[3].lower().replace(' ', '_'), False,True)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/5)*5 - math.floor(min_x/10)*10)/10, int, 0)

    #Plots the feature imporance
    importance(0,5,0, "Local\nImportance:")
    importance(1,5,1, "")
    importance(2,5,2, "")
    importance(3,5,3, "")

    #Plots the density histograms
    axis, min_x, max_x = feat_hist(0, names[0].lower().replace(' ', '_'), True,False)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1,(math.ceil(max_x/10)*10)/10, int, 0)

    axis, min_x, max_x = feat_hist(1, names[1].lower().replace(' ', '_'), False,False)
    plot_limits(axis, min_x, max_x + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

    axis, min_x, max_x = feat_hist(2, names[2].lower().replace(' ', '_'), False,False)
    plot_limits(axis, min_x, max_x + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

    axis, min_x, max_x = feat_hist(3, names[3].lower().replace(' ', '_'), False,True)
    plot_limits(axis, math.floor(min_x/10)*10 - 10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/5)*5 - math.floor(min_x/10)*10)/10, int, 0)

    remove_ax_labels(10)

    #Set the size of the figure
    fig.set_size_inches((8.5, 15), forward=False)
    plt.subplots_adjust(left = 0.012, bottom = 0.070, right = 0.967, top = 0.938, wspace = 0.275, hspace = 0.330)

#    plt.show()
#    continue

    ### SECOND SET OF PLOTS -- INTERACTIONS ###
    fig, ax = plt.subplots(4,8)
    remove_ax_labels(7)

    big_images(False,True)

    ### USEFUL FUNCTIONS ###
    max_cm = max(np.array(ebm.term_scores_[4:8]).flatten())
    min_cm = min(np.array(ebm.term_scores_[4:8]).flatten())
    int_max = max(abs(max_cm), abs(min_cm))

    ### Interactions ###
    def int_image_plots(a, images, title, b):
        if ('warm' in images[0]) | ('cool' in images[0]):
            ax[a,4].imshow(ones, alpha = eval(images[0]), origin = 'lower', cmap = 'gray')
        else:
            ax[a,4].imshow(eval(images[0]), origin = 'lower', cmap = 'gray')

        if ('warm' in images[1]) | ('cool' in images[1]):
            ax[a,5].imshow(ones, alpha = eval(images[1]), origin = 'lower', cmap = 'gray')
        else:
            ax[a,5].imshow(eval(images[1]), origin = 'lower', cmap = 'gray')
        ax[a,4].set_xlabel(title[0])
        ax[a,5].set_xlabel(title[1])
        ax[a,4].set_title(b, x = 1.15)

    ### Interaction Shape Functions ###
    def shape_fx_ints(a,b,labels,title):
        x = np.array(ebm.explain_global().data(b)['left_names'])
        y = np.array(ebm.explain_global().data(b)['right_names'])
        z = np.array(ebm.explain_global().data(b)['scores'].T)

#        if (a == 0):
#            print(x)
#            print(y)
#            for i in range(len(z.flatten())):
#                print(z.flatten()[i], ', ', end =" ")
#                if(i%30 == 0):
#                    print()

        interaction = ax[a,7].pcolormesh(x,y,z, cmap = 'seismic_r', vmin = -plot_max, vmax = plot_max)
        plt.colorbar(interaction, ax = ax[a,7], location = 'right', fraction = 0.046, pad = 0.04)

        #Manages y-ticks
        y_labels    = np.zeros(len(ax[a,7].yaxis.set_ticklabels([])))
        y_positions = np.linspace(min(ebm.explain_global().data(b)['right_names']), max(ebm.explain_global().data(b)['right_names']), len(y_labels))

        y_labels[0] = np.round(y_positions[0],2)
        y_labels[len(y_labels) - 1] = np.round(y_positions[len(y_labels) - 1],2)

        ax[a,7].yaxis.set_major_locator(ticker.FixedLocator(y_positions))
        ax[a,7].yaxis.set_major_formatter(ticker.FixedFormatter(y_labels))

        [label.set_visible(False) for label in ax[a,7].yaxis.get_ticklabels()[1:len(y_labels) - 1]]

        #Manages x-ticks
        x_labels    = np.zeros(len(ax[a,7].xaxis.set_ticklabels([])))
        x_positions = np.linspace(min(ebm.explain_global().data(b)['left_names']), max(ebm.explain_global().data(b)['left_names']), len(x_labels))

        x_labels[0] = np.round(x_positions[0],2)
        x_labels[len(x_labels) - 1] = np.round(x_positions[len(x_labels) - 1],2)

        ax[a,7].xaxis.set_major_locator(ticker.FixedLocator(x_positions))
        ax[a,7].xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))

        [label.set_visible(False) for label in ax[a,7].xaxis.get_ticklabels()[1:len(x_labels) - 1]]

        #Sets axis labels and title
        ax[a,7].set_xlabel(labels[0])
        ax[a,7].set_ylabel(labels[1])
        if 'GLCM' in labels[1]:
            ax[a,7].yaxis.labelpad = -15
        else:
            ax[a,7].yaxis.labelpad = -29
        ax[a,7].xaxis.labelpad = -7
        ax[a,7].set_title(title)
        ax[a,7].tick_params(axis='both', which='major', labelsize=7)

    #Plots the interactions themselves
    int_image_plots(0, names[4].lower().replace(' ', '_').split('_&_'), names[4].replace('&', '').split('  '), "Features:")
    int_image_plots(1, names[5].lower().replace(' ', '_').split('_&_'), names[5].replace('&', '').split('  '), "")
    int_image_plots(2, names[6].lower().replace(' ', '_').split('_&_'), names[6].replace('&', '').split('  '), "")
    int_image_plots(3, names[7].lower().replace(' ', '_').split('_&_'), names[7].replace('&', '').split('  '), "")

    #6,5,4,7
    #Plots the feature importance of the interactions
    importance(0,6,4, "Feature Importance:")
    importance(1,6,5, "")
    importance(2,6,6, "")
    importance(3,6,7, "")

    #Plots the shape functions of the interactions
    shape_fx_ints(0,4, names[4].replace('&', '').split('  '), "Shape Function:")
    shape_fx_ints(1,5, names[5].replace('&', '').split('  '), "")
    shape_fx_ints(2,6, names[6].replace('&', '').split('  '), "")
    shape_fx_ints(3,7, names[7].replace('&', '').split('  '), "")

    #Sets the size of the figure & plots
    fig.set_size_inches((8.5, 11), forward=False)
    plt.subplots_adjust(left = 0.012, bottom = 0.045, right = 0.971, top = 0.938, wspace = 0.2, hspace = 0.245)
    plt.show()

#    print(ebm.term_names_)

#    filepath = r'/home/nmitchell/GLCM/EBM-no-interaction/'
#    filepath += 'EBM_no_interaction_' + str(isamp) + ".png"
#    fig.savefig(filepath)
#    print(filepath)
#    plt.close()

