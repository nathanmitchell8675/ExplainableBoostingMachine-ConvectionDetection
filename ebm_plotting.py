import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import axes
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xarray as xr
import pickle as pkl

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from matplotlib import ticker

import math
import warnings
warnings.filterwarnings('ignore')
#####################################################

tile_size = 4

#New EBM -- brightness & infrared data were *standardized*, warm and cool GLCM were left as-is
#filepath = r'/home/nmitchell/GLCM/models/EBM_model_StandardScaler_noADASYN'
filepath = r'/home/nmitchell/GLCM/models/EBM_model_ADASYN'

### Model Loading ###
with open(filepath, 'rb') as file:
    model = pkl.load(file)
ebm = model["Model"][0]

### Feature Renaming ###
names = ebm.term_names_
names = '  '.join(names)

#print(ebm.term_names_)

feature_names = ['Brightness', 'Warm GLCM', 'Cool GLCM', 'Infrared Image']
for i in range(len(feature_names)):
    names = names.replace('feature_000' + str(i), feature_names[i])

names = names = names.split('  ')

### Intercept Alteration ###
ebm.intercept_ = ebm.intercept_ + 2

### Shape Function Alteration ###
### f(x): Convolved Image (0) ###
#ebm.term_scores_[0][75:113] = 2 + ebm.term_scores_[0][75:113]
#ebm.term_scores_[0][75:113] = np.full(113-75, 0)
#ebm.term_scores_[0][0:70] = -2 + ebm.term_scores_[0][0:70]

### f(x): Warm GLCM (1) ###
#ebm.term_scores_[1] = ebm.term_scores_[1]*2

### f(x): Cool GLCM (2) ###
#x = np.linspace(0,1.45,len(ebm.explain_global().data(2)['scores']))
#y = np.array((np.sin(x)**np.sin(x)*np.tan(x) - 4)/((x**2)+2)+1)

x = np.linspace(0,2.15,len(ebm.explain_global().data(2)['scores']))
y = np.array((x**2) - 1)

#ebm.term_scores_[2] = y
#ebm.explain_global().data(2)['scores'][:] = y

#x = np.linspace(0,8,1024)
#ebm.term_scores_[2] = np.array(2*np.arctan(x-(np.pi/2)))
#ebm.term_scores_[2] = np.array((1/100)*x**3 - 1)

### f(x): Infrared Image (3) ###

### f(x): Interactions (in order) ###
#ebm.term_scores_[4] = np.zeros((32,32))
#ebm.term_scores_[5] = np.zeros((32,32))
#ebm.term_scores_[6] = np.zeros((32,32))
#ebm.term_scores_[7] = np.zeros((32,32))

validation_data = xr.open_dataset('/home/nmitchell/GLCM/validation_data.nc')

ones = np.ones((int(256/tile_size), int(254/tile_size)))
red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["red", "black"])
gt_cm  = mpl.colors.LinearSegmentedColormap.from_list(" ", ["white", "red"])

#505, 509

#EBM: predicts convection fairly well
#examples = [3,10,14,28,49,53,67,71,75,88,90,100,106,110]
#EBM: hallucinates convection (due to GLCM tiles & high, wispy clouds)
#examples = [9,20,22,32,37,48,52,55,61,63,72,76,91,92,80]
#EBM: underpredicts convection (63, 72)
#examples = [17,24,27,66,105,505]
#Low GLCM (typically, non-zero-mean < 0.01) (overpredicts convection)
#examples = [20,37,72,76]
#Overlap Examples:
#20,21

#10
#92
num1 = 10
num2 = 11

examples = [3,10,14,28]

#for isamp in range(num1,num2):
for isamp in examples:
    #Pulls all of the variables needed to create X. X is needed to make predictions.
    #Data Pulled: Convoled Image, GLCM Above Threshold, GLCM Below Threshold, Infrared Image
    brightness = validation_data.Convolved_Image.sel(Sample = isamp).values.reshape(-1,1).flatten()
    infrared   = validation_data.Infrared_Image.sel(Sample = isamp).values.reshape(-1,1).flatten()
    warm_glcm  = validation_data.Above_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(-1,1).flatten()
    cool_glcm  = validation_data.Below_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(-1,1).flatten()

    X_val = np.transpose(np.array([brightness, warm_glcm, cool_glcm, infrared]))
    y_val = validation_data.Masked_Truth.values.flatten()

    #Pull Remaining Images & Reshape
    full_mrms        = validation_data.Ground_Truth.sel(Sample = isamp).values.reshape(int(256/tile_size), -1)
    masked_mrms      = validation_data.Masked_Truth.sel(Sample = isamp).values.reshape(int(256/tile_size), -1).astype(np.uint8)
    brightness       = brightness.reshape(64,64)
    original_image   = validation_data.Original_Image.sel(Sample = isamp).values.reshape(256,256)
    infrared_image   = validation_data.Infrared_Image.sel(Sample = isamp).values.reshape(64,64)

    #Pull the Warm (> 250K) & Cool (< 250K) GLCM Tiles & Reshape
    warm_glcm = validation_data.Above_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(64,64)
    cool_glcm = validation_data.Below_IR_Mask_Applied_to_OG_GLCM.sel(Sample = isamp).values.reshape(64,64)

    #print(round(np.nanmean(np.where(cool_glcm == 0.0, np.nan, cool_glcm)), 5), isamp)

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
    def big_images():

        ### Original Image ###
        og_im_ax = fig.add_subplot(gs[0:2, 0:2])
        og_im_ax.imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        og_im_ax.yaxis.set_label_position("right")
        intercept = "Intercept: " + str(np.round(ebm.intercept_[0], 3))
        og_im_ax.set_ylabel(intercept, rotation = 0, labelpad = 115, fontsize = 15)
        remove_ticks(og_im_ax, "Channel 2: Visible Imagery")
        ###----------------###

        ### Ground Truth ###
        mrms_ax = fig.add_subplot(gs[2:4, 0:2])
        mrms_ax.imshow(original_image, alpha = 0.5,  cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        mrms_ax.imshow(ones, alpha = full_mrms*0.25, cmap = red_cm, origin = 'lower', extent = (0,256,0,256)) #Full Ground Truth (low opacity)
        mrms_ax.imshow(ones, alpha = masked_mrms,    cmap = red_cm, origin = 'lower', extent = (0,256,0,256)) #IR Masked Ground Truth (full opacity)
        remove_ticks(mrms_ax, "Multi-Radar Multi-Sensor (MRMS)")
        ###--------------###

        ### Predicted Convection ###
        predictions       = ebm.predict(X_val)
        pred_convection   = np.array([float(i) for i in predictions]).reshape(int(256/tile_size), int(256/tile_size))
        pred_convection_2 = np.array([float(i) for i in predictions]).reshape(int(256/tile_size), int(256/tile_size))

        N = len(pred_convection)

        def remove_bits():
            for i in range(N):
                for j in range(N):
                    total = int(pred_convection[(i-1)%N,(j-1)%N] + pred_convection[(i-1)%N,j] + pred_convection[(i+1)%N,(j+1)%N] +
                            pred_convection[i,(j-1)%N]                                        + pred_convection[i,(j+1)%N] +
                            pred_convection[(i+1)%N,(j-1)%N]     + pred_convection[(i+1)%N,j] + pred_convection[(i-1)%N,(j+1)%N])

                    if((total <= 3) & (pred_convection[i,j] == 1.)):
                        pred_convection_2[i,j] = 0

        #remove_bits()

        pred_convection_ax = fig.add_subplot(gs[2:4, 2:4])
        pred_convection_ax.imshow(original_image, cmap = 'gray', origin = 'lower', extent = (0,256,0,256))
        pred_convection_ax.imshow(ones, alpha = (np.array(pred_convection_2)*ebm.predict_proba(X_val)[:,1].reshape(64,64)), cmap = red_cm, origin = 'lower', extent = (0,256,0,256))
        remove_ticks(pred_convection_ax, "Predicted Convection")

        #OPTIONAL: only keep predictions above a specific "certainty" threshold
        certainty = 0.75
        #pred_convection_ax.imshow( np.array(np.where(ebm.predict_proba(X_val)[:,1] >= certainty,1,np.nan)).reshape(64,64), cmap = red_cm, origin = 'lower', extent = (0,256,0,256))
        ### ---------------------###


    ### Feature Imporance ###
    max_cm = max(ebm.eval_terms(X_val)[:,0:4].flatten())
    min_cm = min(ebm.eval_terms(X_val)[:,0:4].flatten())
    plot_max = max(abs(max_cm), abs(min_cm))

    def importance(a,b,c, title):
        ax[a,b].set_title(title)
        importance = ax[a,b].imshow(ebm.eval_terms(X_val)[:,c].reshape(64,64), cmap = 'seismic_r', origin = 'lower', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        plt.colorbar(importance, ax = ax[a,b], location = 'bottom', fraction=0.046, pad=0.04)
    ###-------------------###

    ### Shape Functions ###
    def shape_function(a, title, name):
        data = eval(name).flatten()
        shape_fx = fig.add_subplot(gs[a, 6:8])
        shape_fx.grid(True, linestyle = ':')
        shape_fx.set_axisbelow(True)

        #shape_fx.plot(ebm.explain_global().data(a)['names'][0:len(ebm.explain_global().data(a)['names']) - 1], ebm.explain_global().data(a)['scores'])
        #plt.fill_between(ebm.explain_global().data(a)['names'][0:len(ebm.explain_global().data(a)['names']) - 1], ebm.explain_global().data(a)['lower_bounds'], ebm.explain_global().data(a)['upper_bounds'], alpha = 0.5)

        xvals = ebm.explain_global().data(a)['names'][0:len(ebm.explain_global().data(a)['names']) - 1]
        yvals = ebm.explain_global().data(a)['scores']

        min_val = np.nanmin(np.where(data*(masked_mrms.flatten()==1.0) == 0, np.nan, data*(masked_mrms.flatten()==1.0)))
        max_val = np.nanmax(np.where(data*(masked_mrms.flatten()==1.0) == 0, np.nan, data*(masked_mrms.flatten()==1.0)))

        if (math.isnan(min_val)) | (math.isnan(max_val)):
            min_val = np.float64(0)
            max_val = np.float64(0)

        if ('warm' in name) | ('cool' in name):
            add_on = 0.01
        else:
            add_on = 1

        xval_conv = np.where((xvals >= min_val) & (xvals <= max_val), xvals, np.nan)
        xval_nconv = np.where((xvals <= min_val + add_on) | (xvals >= max_val - add_on), xvals, np.nan)

        yval_conv = np.where((xvals >= min_val) & (xvals <= max_val), yvals, np.nan)
        yval_nconv = np.where((xvals <= min_val + add_on) | (xvals >= max_val - add_on), yvals, np.nan)

        upper_conv = np.where((xvals >= min_val) & (xvals <= max_val), ebm.explain_global().data(a)['upper_bounds'], np.nan)
        lower_conv = np.where((xvals >= min_val) & (xvals <= max_val), ebm.explain_global().data(a)['lower_bounds'], np.nan)

        upper_nconv = np.where((xvals <= min_val + add_on) | (xvals >= max_val - add_on), ebm.explain_global().data(a)['upper_bounds'], np.nan)
        lower_nconv = np.where((xvals <= min_val + add_on) | (xvals >= max_val - add_on), ebm.explain_global().data(a)['lower_bounds'], np.nan)

        shape_fx.plot(xval_conv, yval_conv, color = 'crimson')
        shape_fx.plot(xval_nconv, yval_nconv, color = 'dimgray')

        shape_fx.fill_between(xval_conv, lower_conv, upper_conv, alpha = 0.5, color = 'crimson')
        shape_fx.fill_between(xval_nconv, lower_nconv, upper_nconv, alpha = 0.5, color = 'dimgray')

        shape_fx.set_title(title)
    ###-----------------###

    ### Feature Histograms ###
    def feat_hist(a, data, name, title):
        hist = fig.add_subplot(gs[a,8:10])
        hist.grid(True, linestyle = ':')
        hist.set_axisbelow(True)
        bins = [x for x in ebm.explain_global().data(a)['density']['names'] if x <= max(data)]

        if "Cool" in name:
            n_mrms = masked_mrms.flatten()[infrared_image.flatten() <= 250]
            data = data.flatten()[infrared_image.flatten() <= 250]
        elif "Warm" in name:
            n_mrms = masked_mrms.flatten()[infrared_image.flatten() > 250]
            data = data.flatten()[infrared_image.flatten() > 250]
        else:
            n_mrms = masked_mrms

        hist.hist([data*(n_mrms.flatten()==0.0), data*(n_mrms.flatten()==1.0)],
                  bins = bins, histtype='bar',
                  stacked=True, label=['Non-Convective','Convective'], color=['dimgray','crimson'])
        hist.legend()
        hist.tick_params(bottom=True, top=False, left=False, right=True)
        hist.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True)

        hist.set_title(title)

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
    ax[2,4].imshow(ones, alpha = cool_glcm, cmap = red_cm, origin = 'lower', extent = (0,256,0,256))

    #Infrared Image
    ax[3,4].set_ylabel(names[3], labelpad = 0.1)
    ax[3,4].imshow(infrared_image, cmap = 'gray', origin = 'lower', extent = (0,64,0,64))


    ### FUNCTION CALLS ###
    big_images()

    #Plots the shape functions
    shape_function(0, "Shape Function:", names[0].lower().replace(' ', '_'))
    shape_function(1, "", names[1].lower().replace(' ', '_'))
    shape_function(2, "", names[2].lower().replace(' ', '_'))
    shape_function(3, "", names[3].lower().replace(' ', '_'))

    #Plots the feature imporance
    importance(0,5,0, "Importance:")
    importance(1,5,1, "")
    importance(2,5,2, "")
    importance(3,5,3, "")

    #Plots the density histograms
    feat_hist(0, brightness.flatten(), names[0], "Density:")
    feat_hist(1, warm_glcm.flatten(), names[1], '')
    feat_hist(2, cool_glcm.flatten(), names[2], '')
    feat_hist(3, infrared_image.flatten(), names[3], '')

    remove_ax_labels(10)

    #Set the size of the figure
    fig.set_size_inches((8.5, 15), forward=False)
    plt.subplots_adjust(left = 0.012, bottom = 0.045, right = 0.967, top = 0.938, wspace = 0.175, hspace = 0.330)

#    plt.show()
#    continue

    ### SECOND SET OF PLOTS -- INTERACTIONS ###
    fig, ax = plt.subplots(4,8)
    remove_ax_labels(7)

    big_images()

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
        interaction = ax[a,7].imshow(ebm.explain_global().data(b)['scores'], cmap = 'seismic_r', clim = (-int_max, int_max), origin = 'lower',
                                     extent = [min(ebm.explain_global().data(b)['left_names']), max(ebm.explain_global().data(b)['left_names']),
                                               min(ebm.explain_global().data(b)['right_names']), max(ebm.explain_global().data(b)['right_names'])],
                                     aspect = 'auto')
        plt.colorbar(interaction, ax = ax[a,7], location = 'right', fraction=0.046, pad=0.04)

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

    #Plots the feature importance of the interactions
    importance(0,6,6, "Feature Importance:")
    importance(1,6,5, "")
    importance(2,6,4, "")
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

