# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 21:33:58 2016

@author: Chris Pierse
"""

def NN_Slide_Map(slide_map,X,NN_rows,NN_cols,stride = 5, norm = True):
    import numpy as np
    # NN_rows and NN_cols are the dimensions of the image the NN works on.
    # First find the dimensions and reshape the images.
    N_samples = np.size(X,0)
    if len(X.shape)==4:
        f_img_rows,f_img_cols = np.size(X,2), np.size(X,3)
        X = X.reshape(N_samples,f_img_rows,f_img_cols)
    elif len(X.shape)==3:
        f_img_rows,f_img_cols = np.size(X,1), np.size(X,2)
    row_its, col_its = np.int((f_img_rows-NN_rows)/stride), np.int((f_img_cols-NN_cols)/stride)
    
    count_norm = np.zeros([X.shape[1],X.shape[2]])
    for j in range(row_its):
        for k in range(col_its):
            count_norm[(j)*stride:(j)*stride+NN_rows,(k)*stride:(k)*stride+NN_cols] += 1
    count_norm[count_norm==0]=1
    
    X_map = np.zeros(X.shape)
    for i in range(N_samples):
        for j in range(row_its):
            for k in range(col_its):
                X_map[i,(j)*stride:(j)*stride+NN_rows,(k)*stride:(k)*stride+NN_cols] += slide_map[i,j,k]
        if norm: X_map[i,:,:] = np.divide(X_map[i,:,:],count_norm)
    
    #X_mapped_i = X_map[i,:,:]*X[i,:,:]    
    
    return X_map
    
    
