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

import math
import warnings
warnings.filterwarnings('ignore')
#####################################################

#New EBM -- brightness & infrared data were *standardized*, warm and cool GLCM were left as-is
#filepath = r'/home/nmitchell/GLCM/models/EBM_model_StandardScaler_noADASYN'
filepath = r'/home/nmitchell/GLCM/models/EBM_model_ADASYN'

### Model Loading ###
with open(filepath, 'rb') as file:
    model = pkl.load(file)
ebm = model["Model"][0]

#ebm.scale(4,0)

#ebm.scale(0, 1.60)

xvals = np.array(ebm.explain_global().data(2)['names'][1:1023])
yvals = np.array(ebm.explain_global().data(2)['scores'])

#ebm.explain_global().data(2)['scores'][:] = (xvals*4 + yvals/1.5 - 1)*2
#ebm.explain_global().data(2)['scores'][986:1022] = yvals[986:1022]

### f(x): Infrared Image (3) ###

### f(x): Interactions ###
x = np.array(ebm.explain_global().data(7)['left_names'])
y = np.array(ebm.explain_global().data(7)['right_names'])

x_stop = np.sum(x <= 0.025)
y_stop = np.sum(y <= 225)

#(ebm.explain_global().data(7)['scores'].T)[0:y_stop, 0:x_stop] = (((ebm.explain_global().data(7)['scores'].T)[0:y_stop, 0:x_stop])/1) - 1


### Feature Renaming ###
names = ebm.term_names_
names = '  '.join(names)

#print(ebm.term_names_)

feature_names = ['Brightness', 'Warm GLCM', 'Cool GLCM', 'Infrared Image']
for i in range(len(feature_names)):
    names = names.replace('feature_000' + str(i), feature_names[i])

names = names.split('  ')

#print(names)

all_scores, all_names = list(zip(*sorted(zip(ebm.explain_global().data()['scores'], names), reverse = True)))

training_data   = xr.open_dataset('/home/nmitchell/GLCM/training_data.nc')

#Bring in the X variables -- Convolved Images, GLCM Values, and Infrared Images
### TRAINING DATA ###
convolved_data_training  = training_data.Convolved_Image.values.flatten()
infrared_data_training   = training_data.Infrared_Image.values.flatten()
glcm_data_training_warm  = training_data.Above_IR_Mask_Applied_to_OG_GLCM.values.flatten()
glcm_data_training_cool  = training_data.Below_IR_Mask_Applied_to_OG_GLCM.values.flatten()
mrms_training = training_data.Masked_Truth.values.flatten()
#mrms_training = np.where(np.isnan(mrms_training), 0.0, mrms_training)


plot_max = max(8.9, -8.9)


fig, ax = plt.subplots(5,6)
gs = ax[0,0].get_gridspec()

for i in range(0,4):
    for j in range(0,6):
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

ax[4,0].set_visible(False)
ax[4,1].set_visible(False)


#importance_ax = fig.add_subplot(gs[0:4, 0:2])
#importance_ax.grid(True, linestyle = ':', zorder = -1.0, color = 'dimgray')
#importance_ax.barh(all_names, all_scores, color = 'dimgray')

#importance_ax.tick_params(axis='y', which='major', pad=-5, length = 0, labelcolor = 'w')
#importance_ax.set_yticklabels(all_names, ha='left', size = 15)

#importance_ax.xaxis.set_major_locator(ticker.FixedLocator(np.round(np.arange(0, max(all_scores) + 0.5, 0.5).astype(float), 2)))
#importance_ax.xaxis.set_major_formatter(ticker.FixedFormatter(np.round(np.arange(0, max(all_scores) + 0.5, 0.5).astype(float), 2)))
#[label.set_visible(False) for label in importance_ax.xaxis.get_ticklabels()[1::2]]

#for i in range(len(all_names)-1):
#    importance_ax.text(all_scores[i] + 0.01, i , all_names[i], va = 'center', ha = 'left', size = 15)

#[label.set_visible(False) for label in importance_ax.yaxis.get_ticklabels()[:len(all_names)-1:]]

#importance_ax.set_title("Global Weighted Mean Absolute Score")

### SHAPE FUNCTION ###
def shape_function(a, name, first_plot):
    shape_fx = fig.add_subplot(gs[a,2:4])
    shape_fx.grid(True, linestyle = ':')
    shape_fx.set_axisbelow(True)

    xvals = ebm.explain_global().data(a)['names'][1:len(ebm.explain_global().data(a)['names'])]
    yvals = ebm.explain_global().data(a)['scores']

    shape_fx.plot(xvals, yvals, color = 'dimgray')
    shape_fx.fill_between(xvals, ebm.explain_global().data(a)['lower_bounds'], ebm.explain_global().data(a)['upper_bounds'], alpha = 0.5, color = 'dimgray')

    shape_fx.set_ylabel(name, labelpad = 2)

    shape_fx.yaxis.set_major_locator(ticker.FixedLocator(np.arange(np.floor(min(yvals)) - 1, np.ceil(max(yvals)) + 1).astype(int)))
    shape_fx.yaxis.set_major_formatter(ticker.FixedFormatter(np.arange(np.floor(min(yvals)) - 1, np.ceil(max(yvals)) + 1).astype(int)))

    label_text = "Training Data Convection Percentage: " + str(np.round(np.mean(mrms_training)*100,2)) + '%'

    if(first_plot):
        shape_fx.set_title("Global Shape Function:")
        print(label_text)
        shape_fx.text(110,5,label_text)

    return shape_fx, min(xvals), max(xvals)


### FEATURE DISTRIBUTION ###
def global_hist(data, a, name, first_plot):
    data = eval(data)
    hist = fig.add_subplot(gs[a,4:6])
    hist.grid(True, linestyle = ':')
    hist.set_axisbelow(True)

    bins = ebm.explain_global().data(a)['density']['names']

    if "Cool" in name:
        n_mrms = mrms_training[infrared_data_training <= 250]
        data = data[infrared_data_training <= 250]
    elif "Warm" in name:
        n_mrms = mrms_training[infrared_data_training > 250]
        data = data[infrared_data_training > 250]
    else:
        n_mrms = np.array(mrms_training)

    hist.hist([data], bins = 30, histtype='bar', color = 'dimgray')

    if 'Warm' not in name:
        hist.yaxis.set_major_locator(ticker.FixedLocator(np.arange(np.floor(min(hist.get_yticks())), np.ceil(max(hist.get_yticks())) + 1, 1000000).astype(int)))
        hist.yaxis.set_major_formatter(ticker.FixedFormatter(np.arange(np.floor(min(hist.get_yticks())), np.ceil(max(hist.get_yticks())) + 1, 1000000).astype(int)))
    else:
        hist.yaxis.set_major_locator(ticker.FixedLocator(np.arange(np.floor(min(hist.get_yticks())), np.ceil(max(hist.get_yticks())) + 1, 2000000).astype(int)))
        hist.yaxis.set_major_formatter(ticker.FixedFormatter(np.arange(np.floor(min(hist.get_yticks())), np.ceil(max(hist.get_yticks())) + 1, 2000000).astype(int)))

    hist.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    if(first_plot):
        hist.set_title("Global Feature Distribution:")

    return hist, min(bins), max(bins)

### INTERACTION SHAPE FUNCTIONS ###
def shape_fx_ints(a,b,labels, title, Intercept, Title):
    x = ebm.explain_global().data(b)['left_names']
    y = ebm.explain_global().data(b)['right_names']
    z = ebm.explain_global().data(b)['scores'].T

    interaction = ax[4,a].pcolormesh(x,y,z, cmap = 'seismic_r', vmin = -plot_max, vmax = plot_max)
    plt.colorbar(interaction, ax = ax[4,a], location = 'right', fraction = 0.046, pad = 0.04)

    #Manages y-ticks
    y_labels    = np.zeros(len(ax[4,a].yaxis.set_ticklabels([])))
    y_positions = np.linspace(min(ebm.explain_global().data(b)['right_names']), max(ebm.explain_global().data(b)['right_names']), len(y_labels))

    y_labels[0] = np.round(y_positions[0],2)
    y_labels[len(y_labels) - 1] = np.round(y_positions[len(y_labels) - 1],2)

    ax[4,a].yaxis.set_major_locator(ticker.FixedLocator(y_positions))
    ax[4,a].yaxis.set_major_formatter(ticker.FixedFormatter(y_labels))

    [label.set_visible(False) for label in ax[4,a].yaxis.get_ticklabels()[1:len(y_labels) - 1]]

    #Manages x-ticks
    x_labels    = np.zeros(len(ax[4,a].xaxis.set_ticklabels([])))
    x_positions = np.linspace(min(ebm.explain_global().data(b)['left_names']), max(ebm.explain_global().data(b)['left_names']), len(x_labels))

    x_labels[0] = np.round(x_positions[0],2)
    x_labels[len(x_labels) - 1] = np.round(x_positions[len(x_labels) - 1],2)

    ax[4,a].xaxis.set_major_locator(ticker.FixedLocator(x_positions))
    ax[4,a].xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))

    [label.set_visible(False) for label in ax[4,a].xaxis.get_ticklabels()[1:len(x_labels) - 1]]

    #Sets axis labels and title
    ax[4,a].set_xlabel(labels[0], size = 8)
    ax[4,a].set_ylabel(labels[1], size = 8)
    if 'GLCM' in labels[1]:
        ax[4,a].yaxis.labelpad = -15
    else:
        ax[4,a].yaxis.labelpad = -29
    ax[4,a].xaxis.labelpad = -7

    if Intercept:
        x = -3.5
    if Title:
        x = 1.5
    ax[4,a].set_title(title, x = x, y = -0.5)

    ax[4,a].tick_params(axis='both', which='major', labelsize=7)


def plot_limits(ax, lower_x, upper_x, step_x, lab_type, round_by):
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.round(np.arange(lower_x, upper_x, step_x).astype(lab_type), round_by)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(np.round(np.arange(lower_x, upper_x, step_x).astype(lab_type), round_by)))
    [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[1::2]]


#shape_function(0, names[0].replace(' ', '\n'), True)
#shape_function(1, names[1].replace(' ', '\n'), False)
#shape_function(2, names[2].replace(' ', '\n'), False)
#shape_function(3, names[3].replace(' ', '\n'), False)


axis, min_x, max_x = shape_function(0, names[0], True)
plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/10)*10)/10, int, 0)

axis, min_x, max_x = shape_function(1, names[1], False)
plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*1) + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

axis, min_x, max_x = shape_function(2, names[2], False)
plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*1) + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

axis, min_x, max_x = shape_function(3, names[3], False)
plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/5)*5 - math.floor(min_x/10)*10)/10, int, 0)


axis, min_x, max_x = global_hist("convolved_data_training", 0, names[0].replace(' ', '\n'), True)
plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1,(math.ceil(max_x/10)*10)/10, int, 0)

axis, min_x, max_x = global_hist("glcm_data_training_warm", 1, names[1].replace(' ', '\n'), False)
plot_limits(axis, min_x, max_x + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

axis, min_x, max_x = global_hist("glcm_data_training_cool", 2, names[2].replace(' ', '\n'), False)
plot_limits(axis, min_x, max_x + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

axis, min_x, max_x = global_hist("infrared_data_training",  3, names[3].replace(' ', '\n'), False)
plot_limits(axis, math.floor(min_x/10)*10 - 10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/5)*5 - math.floor(min_x/10)*10)/10, int, 0)

#intercept_addition = 2.25
#ebm.intercept_ = ebm.intercept_ + intercept_addition

intercept = np.round(ebm.intercept_[0],3)

intercept_string = "Intercept: " + str(intercept)

shape_fx_ints(2,4, names[4].replace('&', '').split('  '), intercept_string, True, False)
shape_fx_ints(3,5, names[5].replace('&', '').split('  '), "Interaction Global Shape Functions", False, True)
shape_fx_ints(4,6, names[6].replace('&', '').split('  '), "", False, False)
shape_fx_ints(5,7, names[7].replace('&', '').split('  '), "", False, False)


###########33
### Feature Importance ###

data = [["Weighted Mean Absolute Score"]]

for i in range(len(all_names)):
    data.append([all_names[i].replace(' & ', '/\n'), np.round(all_scores[i], 3)])

column_headers = data.pop(0)
row_headers = [x.pop(0) for x in data]

cell_text = []
for row in data:
    cell_text.append([x for x in row])

rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

table_ax = fig.add_subplot(gs[1:4,0:2])

the_table = table_ax.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowColours=rcolors,
                      rowLoc='right',
                      colColours=ccolors,
                      colLabels=column_headers,
                      loc='center',
                      cellLoc = 'center')

the_table.scale(1.25, 2.5)

the_table.auto_set_font_size(False)
the_table.set_fontsize(17.5)

for i in range(0,4):
    for j in range(0,2):
        ax[i,j].set_visible(False)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.box(on=None)

fig.set_size_inches((8.5, 15), forward=False)
plt.subplots_adjust(left = 0.18, bottom = 0.07, right = 0.97, top = 0.90, wspace = 0.85, hspace = 0.33)
plt.show()
