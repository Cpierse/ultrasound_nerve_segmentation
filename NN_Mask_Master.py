# -*- coding: utf-8 -*-
"""
Created on Sat Jul 09 18:13:48 2016
Iterations on fitting and predicting with the CNN.

This code trains multiple neural networks and combines their outputs. The 
output of a single Neural network is a mask containing probabilities that each
region contains the ROI.
@author: Chris Pierse
"""
# Basic Imports
from __future__ import print_function
import numpy as np
import os, sys
import My_NNs
import matplotlib.pyplot as plt
np.random.seed(1337)
from keras.models import model_from_json

# Import necessary Scripts:
import load_process_data


#%% Fit the U-Net:
folder = 'Neural_Net_Full'
f_rows,f_cols = 64,80

# Data and masks:
X,Y,names = load_process_data.load_data(f_rows,f_cols, file_dir = 'train')
N_samples = X.shape[0]
plt.imshow(X[0,0,:,:],cmap='gray')
plt.imshow(Y[0,0,:,:],cmap='gray')

# Number of test data points:
n_test = np.int(round(N_samples/5))

# The Neural Net:
#t_idxs = np.random.permutation(N_samples)
test_idxs=[] # t_idxs[0:n_test]
train_idxs=range(N_samples) #  t_idxs[n_test:N_samples]
from keras.optimizers import Adam
[model, score] = My_NNs.unet(X,Y,train_idxs,test_idxs,optomizer=Adam(lr=1e-5),batch_size=64,nb_epoch=30)
json_string = model.to_json()
open(os.path.join(folder, 'Unet_' + str(f_rows) + 'x' + str(f_cols) + '_model.json'), 'w').write(json_string)
model.save_weights(os.path.join(folder, 'Unet_' + str(f_rows) + 'x' + str(f_cols) + '_weights.h5'),overwrite=True)

#%% Predict with U-Net:
folder = 'Neural_Net_Full'
f_rows,f_cols = 64,80
model_loc_name = os.path.join(folder, 'Unet_' + str(f_rows) + 'x' + str(f_cols) )
#model_loc_name = os.path.join(folder, 'Unet_T_' + str(f_rows) + 'x' + str(f_cols) )


# Load the test set:
X_test,empty,names = load_process_data.load_data(f_rows,f_cols, file_dir = 'test')
N_samples = X_test.shape[0]
plt.imshow(X_test[0,0,:,:],cmap='gray'); plt.show()

# Add in saliency map - made it worse:
import make_saliency_map
X_test = np.append(X_test,make_saliency_map.saliency_maps(X_test),axis=1)

# Load the model:
model = model_from_json(open(model_loc_name+'_model.json').read())
model.load_weights(model_loc_name + '_weights.h5')

Y_pred = model.predict(X_test)
for i in range(0,5):
    plt.imshow(X_test[i,0,:,:],cmap='gray'); plt.show()
    plt.imshow(Y_pred[i,0,:,:],cmap='gray'); plt.show()

load_process_data.make_submission(Y_pred,names)

#%% Experiment with the U-Net:
folder = 'Neural_Network_Full'
f_rows,f_cols = 64,80

mode = 'test'
# Data and masks:
X,Y,names = load_process_data.load_data(f_rows,f_cols, file_dir = mode)
N_samples = X.shape[0]
plt.imshow(X[0,0,:,:],cmap='gray')
if mode=='train': plt.imshow(Y[0,0,:,:],cmap='gray')

## Add in saliency map - made it worse:
#import make_saliency_map
#X = np.append(X,make_saliency_map.saliency_maps(X),axis=1)

NN_rows, NN_cols = 27,23
f_img_rows, f_img_cols = 84, 116
# Load prob maps made with these parameters
slide_prob_multi = np.load(os.path.join('Neural_Network_Crops',mode + '_slide_prob_multi.npy'))
slide_prob_multi = slide_prob_multi.transpose(1,0,2,3)
# Create the map onto X
X_map = np.zeros([slide_prob_multi.shape[0],slide_prob_multi.shape[1],f_img_rows,f_img_cols])
import NN_Slide_Map
for i in range(X_map.shape[1]):
    X_map[:,i,:,:] = NN_Slide_Map.NN_Slide_Map(slide_prob_multi[:,i,:,:],X_map[:,i,:,:],NN_rows,NN_cols)

import cv2
X_map_resized = np.zeros([X_map.shape[0],X_map.shape[1],X.shape[2],X.shape[3]])
for i in range(X_map.shape[0]):
    for j in range(X_map.shape[1]):
        X_map_resized[i,j,:,:] = cv2.resize(X_map[i,j,:,:], (X.shape[3],X.shape[2]), interpolation = cv2.INTER_AREA)
        

X = np.append(X,X_map_resized,axis=1)

# The Neural Net:
if mode == 'train':
    #t_idxs = np.random.permutation(N_samples)
    test_idxs=[] # t_idxs[0:n_test]
    train_idxs=range(N_samples) #  t_idxs[n_test:N_samples]
    from keras.optimizers import Adam
    [model, score] = My_NNs.unet(X,Y,train_idxs,test_idxs,optomizer=Adam(lr=1e-5),batch_size=64,nb_epoch=30)
    json_string = model.to_json()
    open(os.path.join(folder, 'Unet_T_' + str(f_rows) + 'x' + str(f_cols) + '_model.json'), 'w').write(json_string)
    model.save_weights(os.path.join(folder, 'Unet_T_' + str(f_rows) + 'x' + str(f_cols) + '_weights.h5'),overwrite=True)
elif mode == 'test':
    # Load the model:
    model = model_from_json(open(os.path.join(folder, 'Unet_T_' + str(f_rows) + 'x' + str(f_cols) + '_model.json')).read())
    model.load_weights(os.path.join(folder, 'Unet_T_' + str(f_rows) + 'x' + str(f_cols) + '_weights.h5'))
    
    Y_pred = model.predict(X)
    for i in range(0,5):
        plt.imshow(X[i,0,:,:],cmap='gray'); plt.show()
        plt.imshow(Y_pred[i,0,:,:],cmap='gray'); plt.show()
    
    load_process_data.make_submission(Y_pred,names)

#%% Experiment with data modifications:
folder = 'Neural_Network_Full'
f_rows,f_cols = 64,80

# Data and masks:
X,Y,names = load_process_data.load_data(f_rows,f_cols, file_dir = 'train')
N_samples = X.shape[0]
plt.imshow(X[0,0,:,:],cmap='gray')
plt.imshow(Y[0,0,:,:],cmap='gray')

# Flip the data horizonal, vertical, and both:
X_flipped = np.zeros([N_samples*3,1,f_rows,f_cols])
Y_flipped = np.zeros([N_samples*3,1,f_rows,f_cols])
for i in range(N_samples):
    X_flipped[i,0,:,:] = cv2.flip(X[i,0,:,:],1)
    Y_flipped[i,0,:,:] = cv2.flip(Y[i,0,:,:],1)
    X_flipped[N_samples+i,0,:,:] = cv2.flip(X[i,0,:,:],-1)
    Y_flipped[N_samples+i,0,:,:] = cv2.flip(Y[i,0,:,:],-1)
    X_flipped[2*N_samples+i,0,:,:] = cv2.flip(X[i,0,:,:],0)
    Y_flipped[2*N_samples+i,0,:,:] = cv2.flip(Y[i,0,:,:],0)
plt.imshow(Y_flipped[0,0,:,:],cmap='gray')
plt.imshow(Y_flipped[N_samples,0,:,:],cmap='gray')
plt.imshow(Y_flipped[2*N_samples,0,:,:],cmap='gray')

# Add the new data:
X = np.append(X,X_flipped,axis=0)
Y = np.append(Y,Y_flipped,axis=0)

# Train the CNN:
(test_idxs,train_idxs)=([],range(4*N_samples))
from keras.optimizers import Adam
[model, score] = My_NNs.unet(X,Y,train_idxs,test_idxs,optomizer=Adam(lr=1e-5),batch_size=64,nb_epoch=30)
json_string = model.to_json()
open(os.path.join(folder, 'Unet_flip_' + str(f_rows) + 'x' + str(f_cols) + '_model.json'), 'w').write(json_string)
model.save_weights(os.path.join(folder, 'Unet_flip_' + str(f_rows) + 'x' + str(f_cols) + '_weights.h5'),overwrite=True)


## Test the CNN:
# Load the test data
X_test,empty,names = load_process_data.load_data(f_rows,f_cols, file_dir = 'test')
N_samples = X_test.shape[0]
plt.imshow(X_test[0,0,:,:],cmap='gray'); plt.show()
# Predict
Y_pred = model.predict(X_test)
for i in range(0,5):
    plt.imshow(X_test[i,0,:,:],cmap='gray'); plt.show()
    plt.imshow(Y_pred[i,0,:,:],cmap='gray'); plt.show()
# Save results
load_process_data.make_submission(Y_pred,names)



