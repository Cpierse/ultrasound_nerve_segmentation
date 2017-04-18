# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:57:58 2016
The neural network master.
Here we can define NN parameters and easily train multiple Neural Networks.

This code trains multiple neural networks and combines their outputs. The 
output of a single Neural network is the probability that a particular crop 
contains the ROI. 
@author: Xenogearcap
"""
from __future__ import print_function
import numpy as np
#from PIL import Image
import os
import My_NNs
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

#%% Key variables for this file. Edit as needed. ##
img_rows, img_cols =  27,23 #29, 21 #43, 29 #84, 116 for full
ext = 2;

method = 'random'
folder = 'Neural_Network_Crops'
#%% Load the data as numpy arrays: ###
# This loads a csv with the image intensities from Matlab. 
Y_mast = np.genfromtxt(os.path.join(folder,'Imgs_Targets.csv'), delimiter=',')
N_samples = np.size(Y_mast,0)
mid = ext; tot = 2*ext+1;

# Identify positives to use across all data sets using layer 0:
bp_idxs = np.where(Y_mast==1)[0]
bp_num = len(bp_idxs)
not_bp_idxs = [x for x in range(0,N_samples) if x not in bp_idxs]
not_bp_num = len(not_bp_idxs)
X = np.genfromtxt(  os.path.join(folder,'Imgs_' + str(ext) + str(ext) + '_' + str(img_rows) + 'x' + str(img_cols) + '.csv'), delimiter=',')
X_pos_bp = np.zeros([1,bp_num])
X_neg_bp = np.zeros([1,not_bp_num])

# From layers 1:ex, collect the data:
X_neg_list = []
X_start_idx = np.zeros([tot,tot])
for i in range(0,ext):
    X_neg_list.append(np.empty([0,img_rows*img_cols]))
layer = []
for i in range(0,tot):
    for j in range(0,tot):
        X = np.genfromtxt( os.path.join(folder,'Imgs_' + str(i+1) + str(j+1) + '_' + str(img_rows) + 'x' + str(img_cols) + '.csv'), delimiter=',')
        if len(X.shape)<2: continue        
        layer = np.max( [np.abs(ext-i),np.abs(ext-j)])
        print(layer)
        if layer == 0:
            X_pos_bp = X[bp_idxs,:]
            X_neg_bp = X[not_bp_idxs,:]
            X_start_idx[i,j] = 0
        else:
            X_start_idx[i,j] = X_neg_list[layer-1].shape[0]
            X_neg_list[layer-1] = np.append(X_neg_list[layer-1],X, axis=0)
X_pos_bp = X_pos_bp.reshape(X_pos_bp.shape[0],img_cols,img_rows) # Note: rotated compared to Matlab.
X_pos_bp = X_pos_bp.transpose(0,2,1)
X_neg_bp = X_neg_bp.reshape(X_neg_bp.shape[0],img_cols,img_rows) # Note: rotated compared to Matlab.
X_neg_bp = X_neg_bp.transpose(0,2,1)
for i,X_neg in enumerate(X_neg_list):
    X_neg = X_neg.reshape(X_neg.shape[0],img_cols,img_rows) # Note: rotated compared to Matlab.
    X_neg = X_neg.transpose(0,2,1)
    # clean up any all zero data:
    empty_idx = []
    for idx in range(X_neg.shape[0]):
        if np.sum(X_neg[idx,:,:])==0:
            empty_idx.append(idx)
    X_neg = np.delete(X_neg,empty_idx,axis=0)
    X_neg_list[i] = X_neg
    # Plot an example:
    plt.imshow(X_neg_list[i][0,:,:],cmap='gray')
    plt.show()
# Plot a positive example:
plt.imshow(X_pos_bp[0,:,:],cmap='gray')
plt.show()


## Weird results:
#plt.imshow(X_neg_list[0][0,:,:],cmap='gray')
#plt.imshow(X_pos_bp[0,:,:],cmap='gray')
#
#plt.imshow(X_neg_list[0][2,:,:],cmap='gray')
#plt.imshow(X_pos_bp[1,:,:],cmap='gray')

#%% Training the nets:
def choose_indices(iteration,X_neg,method='random',N=not_bp_num):
    if method == 'sequence':
        idxs = range(iteration*N,(iteration+1)*N)
    elif method == 'random':
        idxs = np.random.permutation(range(X_neg.shape[0]))[0:N]
    Xn = X_neg[idxs,:,:]
    return Xn
    
n_test = np.int(round(N_samples/5))
n_test = 0



layers = range(0,ext+1)
iterations = [1] + [np.int(np.ceil(np.float(x.shape[0])/not_bp_num)) for x in X_neg_list]
scores = np.zeros([ext+1,np.max(iterations),4])
for layer in layers:
    for it in range (0,iterations[layer]):
        Y = np.zeros([N_samples,1])
        if layer == 0:
            X_neg = X_neg_bp
        else:
            X_neg = choose_indices(it,X_neg_list[layer-1])
        X = np.append(X_pos_bp,X_neg,axis=0)
        idxs = np.random.permutation(range(X.shape[0]))
        X = X[idxs]
        Y[idxs<bp_num] = 1
        t_idxs = np.random.permutation(N_samples)
        test_idxs = t_idxs[0:n_test]
        train_idxs = t_idxs[n_test:N_samples]
        [model, score] = My_NNs.CNN(X,Y,train_idxs,test_idxs,nb_filters = 64,opt='adagrad')#,opt='sgd')
        if n_test>0: scores[layer,it,:] = score
        json_string = model.to_json()
        open(os.path.join(folder, 'CNN_' + str(layer) + '-' + str(it) + '_64f_model.json'), 'w').write(json_string)
        model.save_weights(os.path.join(folder, 'CNN_' + str(layer) + '-' + str(it) + '_64f_weights.h5'),overwrite=True)
print(scores)


#%% Training a NN on full images:
f_img_rows, f_img_cols = 84, 116 #122, 140 #84, 116 unrotated data.
folder = 'Neural_Net_Full'

# Load the csv with the image intensities from Matlab. 
# Classifier:
Y = np.genfromtxt('Imgs_Targets.csv', delimiter=',')

N_samples = np.size(Y,0)
X = np.genfromtxt( 'Full_Imgs_' + str(f_img_rows) + 'x' + str(f_img_cols) + '.csv', delimiter=',')

# Reshape the array:
X = X.reshape(X.shape[0],f_img_cols,f_img_rows) # Note: rotated compared to Matlab.
X = X.transpose(0,2,1)
plt.imshow(X[0,:,:],cmap='gray')

# Number of test data points:
n_test = np.int(round(N_samples/5))

# The Neural Net:
idxs = np.random.permutation(range(X.shape[0]))
X,Y = X[idxs],Y[idxs]

t_idxs = np.random.permutation(N_samples)
test_idxs = t_idxs[0:n_test]
train_idxs = t_idxs[n_test:N_samples]
[model, score] = My_NNs.CNN(X,Y,train_idxs,test_idxs,nb_pool = 4,nb_filters = 32,nb_conv = 24,opt='adagrad',activation='tanh',loss='poisson')



#%% Training a NN on full image and masks:
import load_process_data
folder = 'Neural_Net_Full'
f_rows,f_cols = 64,80

# Data and masks:
X,Y = load_process_data.load_data(f_rows,f_cols, file_dir = 'train')
N_samples = Y.shape[0]
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



# Alternative data:
f_rows,f_cols = 80,112
X,Y = load_process_data.load_matlab_csv(f_rows,f_cols)
N_samples = Y.shape[0]
plt.imshow(X[0,0,:,:],cmap='gray')
plt.imshow(Y[0,0,:,:],cmap='gray')

[model, score] = My_NNs.unet(X,Y,train_idxs,test_idxs,optomizer=Adam(lr=1e-5),batch_size=64,nb_epoch=30)
json_string = model.to_json()
open(os.path.join(folder, 'Unet_BF_' + str(f_rows) + 'x' + str(f_cols) + '_model.json'), 'w').write(json_string)
model.save_weights(os.path.join(folder, 'Unet_BF_' + str(f_rows) + 'x' + str(f_cols) + '_weights.h5'),overwrite=True)


