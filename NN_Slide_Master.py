# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 13:13:07 2016
Using a trained Neural Network, slide the network across the input images.
Returns matrices of probability.
@author: Xenogearcap
"""
from __future__ import print_function
import numpy as np
import matplotlib.pylab as plt
import time
import os
from keras.models import model_from_json
import NN_Slider
np.random.seed(1337)  # for reproducibility

#%% Key variables for this file. Edit as needed. ###
NN_rows, NN_cols = 27,23 #29, 21 #43, 29
f_img_rows, f_img_cols = 84, 116 #122, 140 #84, 116 unrotated data.
stride = 5

ext = 2
full_folder = 'Neural_Network_Full'
crop_folder = 'Neural_Network_Crops'

mode = 'test'
#%% Load the data as numpy arrays: ###
# This loads a csv with the full image intensities from Matlab. 
X = np.genfromtxt(os.path.join(full_folder,'Full_Imgs_' + mode + '_' +  str(f_img_rows) + 'x' + str(f_img_cols) + '.csv'), delimiter=',')
# Load targets and resize X.
if mode == 'train':
    Y = np.genfromtxt(os.path.join(crop_folder,'Imgs_Targets.csv'), delimiter=',')
N_samples = np.size(X,0)
X = X.reshape(N_samples,f_img_cols,f_img_rows) # Note: rotated compared to Matlab.
X = X.transpose(0,2,1)

# Extension terms:
mid = ext; tot = 2*ext+1;
# Test an image:
plt.imshow(X[0,:,:],cmap='gray')

#%% The Slide function
row_its, col_its = np.int((f_img_rows-NN_rows)/stride), np.int((f_img_cols-NN_cols)/stride)
slide_class_multi, slide_prob_multi = [],[]
for layer in range(0,ext+1):
    it = 0
    while True:
        # Load the designated model:            
        try:        
            model = model_from_json(open(os.path.join(crop_folder, 'CNN_' + str(layer) + '-' + str(it) + '_64f_model.json')).read())
            model.load_weights(os.path.join(crop_folder, 'CNN_' + str(layer) + '-' + str(it) + '_64f_weights.h5'))
        except: break
        print('Layer ' + str(layer) + ' Iteration ' + str(it) + ' Start')
        # Determine the slide class and slide prob
        slide_class_n, slide_prob_n = NN_Slider.NN_Slide(model,X,NN_rows,NN_cols,method = 1,stride=stride)
        slide_class_multi.append(slide_class_n)
        slide_prob_multi.append(slide_prob_n)
        #plt.imshow(slide_class_n[0,:,:]); plt.show();
        #plt.imshow(slide_prob_n[0,:,:]); plt.show();
        print('Layer ' + str(layer) + ' Iteration ' + str(it) + ' Complete')
        it += 1

np.save(os.path.join(crop_folder,mode + '_slide_class_multi.npy'),np.array(slide_class_multi))
np.save(os.path.join(crop_folder,mode + '_slide_prob_multi.npy'),np.array(slide_prob_multi))
#%% Convert probability map in slide space to image space:
NN_rows, NN_cols = 27,23
f_img_rows, f_img_cols = 84, 116
# Load prob maps made with these parameters
slide_prob_multi = np.load(os.path.join(crop_folder,mode + '_slide_prob_multi.npy'))
slide_prob_multi = slide_prob_multi.transpose(1,0,2,3)
# Create the map onto X
X_map = np.zeros([slide_prob_multi.shape[0],slide_prob_multi.shape[1],f_img_rows,f_img_cols])
import NN_Slide_Map
for i in range(X_map.shape[1]):
    X_map[:,i,:,:] = NN_Slide_Map.NN_Slide_Map(slide_prob_multi[:,i,:,:],X_map[:,i,:,:],NN_rows,NN_cols)

for i in range(0,10):
    plt.imshow(X_map[0,i,:,:],cmap='gray'); plt.show()

#X_map_scaled = np.array(np.round(X_map*255),'uint8')
#import load_process_data
#load_process_data.convert_to_csv(X_map_scaled,mode + '_slide_prob_multi')



