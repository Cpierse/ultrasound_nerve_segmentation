# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:57:58 2016
The neural network master.
Here we can define NN parameters and easily train a Neural Network.

This code trains ONE neural network. To train multiple networks, use 
NN_Mult_Master. The output of a single Neural network is the probability
that a particular crop contains the ROI. 
@author: Xenogearcap
"""
from __future__ import print_function
import numpy as np
from PIL import Image

import My_NNs
np.random.seed(1337)  # for reproducibility

#%% Key variables for this file. Edit as needed. ##
full = 0 # Full image NN or highlighted box?
img_rows, img_cols =  27,23 #29, 21 #43, 29 #84, 116 for full
ex = 0


#%% Load the data as numpy arrays: ###
# This loads a csv with the image intensities from Matlab. 
if full:
    X = np.genfromtxt( 'Full_Imgs_' + str(img_rows) + 'x' + str(img_cols) + '.csv', delimiter=',')
    Y = np.genfromtxt('Imgs_Targets.csv', delimiter=',')
elif ex==1:
    X = np.genfromtxt( 'Imgs_ex_' + str(img_rows) + 'x' + str(img_cols) + '.csv', delimiter=',')
    Y = np.genfromtxt('Imgs_ex_Targets.csv', delimiter=',')
else:
    X = np.genfromtxt( 'Imgs_' + str(img_rows) + 'x' + str(img_cols) + '.csv', delimiter=',')
    Y = np.genfromtxt('Imgs_Targets.csv', delimiter=',')

# Load targets and resize X.
N_samples = np.size(X,0)
X = X.reshape(N_samples,img_cols,img_rows) # Note: rotated compared to Matlab.
X = X.transpose(0,2,1)

# Test an image:
im = Image.fromarray(X[0,:,:])
im.show()

#%% Set up n-fold cross-validation and run the fits: ###
idxs = np.random.permutation(N_samples)
n_fold = 5
cv_samples = np.int(N_samples/n_fold)

# loss = 'binary_crossentropy','kullback_leibler_divergence','poisson'
# final_activation = 'softplus'(1x:0.909), 'sigmoid'(1x:0.913), 'hard_sigmoid'(1x:0.871)
scores = np.zeros([n_fold,4])
for i in range(0,n_fold):
    cvii = range(i*cv_samples,(i+1)*cv_samples)
    test_idxs = idxs[cvii]
    train_idxs = np.delete(idxs, cvii)
    [model, score] = My_NNs.CNN(X,Y,train_idxs,test_idxs,loss='poisson',final_activation='sigmoid')
    scores[i,:] = score
print(scores)
print(np.mean(scores,0))

#%% If the parameters are sufficient, train a single model on everything:
# Train on the entire data set:
train_idxs = range(N_samples)
[model, score] = My_NNs.CNN(X,Y,train_idxs,[])

# Save the results:
if ex==1:
    json_string = model.to_json()
    open('CNN_model_arch_ex_1.json', 'w').write(json_string)
    model.save_weights('CNN_weights_ex_1.h5')
else:
    json_string = model.to_json()
    open('CNN_model_arch_1.json', 'w').write(json_string)
    model.save_weights('CNN_weights_1.h5')







