# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 13:13:07 2016
Using a trained Neural Network, slide the network across the input images.
Returns matrices of probability.
@author: Xenogearcap
"""
def NN_Slide(model,X,NN_rows,NN_cols,method=0,stride = 5):
    import numpy as np
    # NN_rows and NN_cols are the dimensions of the image the NN works on.
    # First find the dimensions and reshape the images.
    N_samples = np.size(X,0)
    if len(X.shape)==4:
        f_img_rows,f_img_cols = np.size(X,2), np.size(X,3)
    elif len(X.shape)==3:
        f_img_rows,f_img_cols = np.size(X,1), np.size(X,2)
        X = X.reshape(N_samples,1,f_img_rows,f_img_cols)
    row_its, col_its = np.int((f_img_rows-NN_rows)/stride), np.int((f_img_cols-NN_cols)/stride)
    
    if method==0:
        # Batch Method - SLower but not memory intensive:
        #Set up slider output:
        slide_class = np.zeros([N_samples,row_its,col_its]);
        slide_prob = np.zeros([N_samples,row_its,col_its]);
        # Iterate over every image:
        X_now = np.zeros([row_its*col_its,1,NN_rows,NN_cols])
        for i in range(N_samples):
            for j in range(row_its):
                for k in range(col_its):
                    X_now[(j)*col_its+k,0,:,:] = X[i,0,(j)*stride:(j)*stride+NN_rows,(k)*stride:(k)*stride+NN_cols]
            X_now = X_now.astype('float32')
            X_now /= 255
            classes = model.predict_classes(X_now, batch_size=row_its*col_its,verbose=0)
            prob = model.predict_proba(X_now, batch_size=row_its*col_its,verbose=0)
            slide_class[i,:,:] = classes.reshape([row_its,col_its])
            slide_prob[i,:,:] = prob.reshape([row_its,col_its])
    elif method==1:    
        # Bulk Method - Fast but memory intensive
        X_ex = np.zeros([N_samples*row_its*col_its,1,NN_rows,NN_cols])
        for i in range(N_samples):
            for j in range(row_its):
                for k in range(col_its):
                    X_ex[(i)*row_its*col_its+(j)*col_its+k,0,:,:] = X[i,0,(j)*stride:(j)*stride+NN_rows,(k)*stride:(k)*stride+NN_cols]
        X_ex = X_ex.astype('float32')
        X_ex /= 255
        slide_class = model.predict_classes(X_ex, batch_size=row_its*col_its,verbose=2)
        slide_prob = model.predict_proba(X_ex, batch_size=row_its*col_its,verbose=2)
        slide_class = slide_class.reshape([N_samples,row_its,col_its])
        slide_prob = slide_prob.reshape([N_samples,row_its,col_its])
    
    return slide_class, slide_prob
