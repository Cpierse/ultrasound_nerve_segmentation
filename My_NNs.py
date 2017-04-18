# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:59:25 2016
Based off of First_CNN.py
Here we create a function to build a CNN given input parameters.
Inputs are:
    X = Image numpy array of shape (N_samples,img_rows,img_cols)
    Y = Targets of size (N_samples,1)
    train_idxs,test_idxs = indexes of the train and validation (test) samples
    nb_filters =n umber of convolutional filters to use
    nb_pool = size of pooling area for max pooling
    nb_conv = convolution kernel size
    nb_epoch = number of passes through entire training set.
@author: Xenogearcap
"""
from __future__ import print_function

#%% Functions:
def process_XY(X,Y,train_idxs,test_idxs,mask=False):
    # import numpy as np
    # Input image dimensions, need to add the 1 for CNN.
    if len(X.shape)==3:
        N_samples,img_rows,img_cols = X.shape
        X = X.reshape([X.shape[0],1,X.shape[1],X.shape[2]])
        channels = 1
    elif len(X.shape)==4:
        N_samples,channels,img_rows,img_cols = X.shape
        if mask: Y = Y.reshape(N_samples,1,img_rows,img_cols)
    #pix = img_rows*img_cols
    
    # Split the data:
    X_train, X_test = X[train_idxs,:,:,:], X[test_idxs,:,:,:]
    if mask: Y_train, Y_test = Y[train_idxs,:,:,:], Y[test_idxs,:,:,:]    
    else: Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    
    # Clean up the data and convert for GPU:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, X_test, Y_train, Y_test, N_samples, channels, img_rows, img_cols

#%% A Basic Convolutional Neural Net
def CNN(X,Y,train_idxs,test_idxs,nb_filters = 32,nb_pool = 2,nb_conv = 3,nb_epoch = 20, nb_dense = 128, loss='poisson',final_activation='sigmoid', opt = 'adadelta', activation = 'relu'):
    # Import the necessay 
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten#, LeakyReLU
    from keras.layers import Convolution2D, MaxPooling2D
    np.random.seed(1337) # For reproducibility:
    from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
    
    # Activation note:
    # To use a LeakyReLU, firse 'from keras.layers import LeakyReLU' then
    # set activation = LeakyReLU(alpha=0.3)
    
    
    # Other parameters that are not worth changing for our data.
    batch_size = 128
    nb_classes = 1
    
    # Process data:
    X_train, X_test, Y_train, Y_test, N_samples, channels, img_rows, img_cols = process_XY(X,Y,train_idxs,test_idxs)
    
    # The model:
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(nb_dense))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation(final_activation))
    
    model.compile(loss=loss,
                  optimizer= opt,
                  metrics=['accuracy'])
                  
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=2, validation_data=(X_test, Y_test),shuffle=True)
    import time
    time.sleep(1)
    if len(X_test)>0:
        y_pred = model.predict_classes(X_test)
        y_pred = y_pred[:,0]
        accuracy = accuracy_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        print('Accuracy: {}'.format(accuracy))
        print('Recall: {}'.format(recall))
        print('Precision: {}'.format(precision))
        print('F1: {}'.format(f1))
        score = np.array([accuracy, recall, precision, f1])
    else:
        score = []
    return model, score

#%% Merging two convolutional neural nets before the dense layer:
def merged_CNN(X1,X2,Y,train_idxs,test_idxs,nb_filters = 32,nb_pool = 2,nb_conv = 3,nb_epoch = 20, nb_dense = 128, loss='poisson',final_activation='sigmoid'):
    # Take two simultaneous inputs X1 and X2 of the same size    
    # Import the necessay 
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, Merge
    from keras.layers import Convolution2D, MaxPooling2D
    np.random.seed(1337) # For reproducibility:
    from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
    
    # Other parameters that are not worth changing for our data.
    batch_size = 128
    nb_classes = 1
    
    # Process data:
    X1_train, X1_test, Y_train, Y_test, N_samples, img_rows, img_cols = process_XY(X1,Y,train_idxs,test_idxs)
    X2_train, X2_test, Y_train, Y_test, N_samples, img_rows, img_cols = process_XY(X2,Y,train_idxs,test_idxs)

    
    # First, the left branch, This is to be trained on the main crop.
    left_branch = Sequential()
    left_branch.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    left_branch.add(Activation('relu'))
    left_branch.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    left_branch.add(Activation('relu'))
    left_branch.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    left_branch.add(Dropout(0.25))
    left_branch.add(Flatten())

    # Then, the right branch, This is to be trained on an altered version of the
    # same crop. For example, on the gradient.
    right_branch = Sequential()
    right_branch.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    right_branch.add(Activation('relu'))
    right_branch.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    right_branch.add(Activation('relu'))
    right_branch.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    right_branch.add(Dropout(0.25))
    right_branch.add(Flatten())

    merged = Merge([left_branch, right_branch], mode='concat')    
    
    model = Sequential()
    model.add(merged)
    model.add(Dense(nb_dense))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation(final_activation))
    
    model.compile(loss=loss,
                  optimizer='adadelta',
                  metrics=['accuracy'])
                  
    model.fit([X1_train,X2_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=2, validation_data=([X1_test,X2_test], Y_test))
    import time
    time.sleep(1)
    if len(X1_test)>0:
        y_pred = model.predict_classes([X1_test,X2_test])
        y_pred = y_pred[:,0]
        accuracy = accuracy_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        print('Accuracy: {}'.format(accuracy))
        print('Recall: {}'.format(recall))
        print('Precision: {}'.format(precision))
        print('F1: {}'.format(f1))
        score = np.array([accuracy, recall, precision, f1])
    else:
        score = []
    return model, score

#%% CNNs whose goal is to return the mask:
# To handle the mask and evaluate based on dice coefficient:
from keras import backend as K
epsilon = float(0.1)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + epsilon) / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#%% U-Net:
def unet(X,Y,train_idxs,test_idxs,nb_epoch=20,optomizer='adagrad',batch_size = 32):
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
    #from keras.optimizers import Adam
    np.random.seed(2016) # For reproducibility:
    #from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
    

    
    # Process data:
    X_train, X_test, Y_train, Y_test, N_samples, channels, img_rows, img_cols = process_XY(X,Y,train_idxs,test_idxs,mask=True)
    
    #inputs = Input((1, X.shape[0], img_cols))
    inputs = Input((channels, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=optomizer, loss=dice_coef_loss, metrics=[dice_coef])#Adam(lr=1e-5)
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test),shuffle=True)
              
    import time
    time.sleep(1)
    if len(X_test)>0:
        score =[]
        Y_pred = model.predict(X_test);
        dc = dice_coef(Y_test, Y_pred);
        score = dc
        print('Dice coefficient: ' + str(dc))
    #        y_pred = model.predict(X_test)
    #        y_pred = y_pred[:,0]
    #        accuracy = accuracy_score(Y_test, y_pred)
    #        recall = recall_score(Y_test, y_pred)
    #        precision = precision_score(Y_test, y_pred)
    #        f1 = f1_score(Y_test, y_pred)
    #        print('Accuracy: {}'.format(accuracy))
    #        print('Recall: {}'.format(recall))
    #        print('Precision: {}'.format(precision))
    #        print('F1: {}'.format(f1))
    #        score = np.array([accuracy, recall, precision, f1])
    else:
        score = []
    return model, score






#%% Another try:
def mask_CNN(X,Y,train_idxs,test_idxs,nb_filters = 32,nb_pool = 2,nb_conv = 3,nb_epoch = 20, nb_dense = 128, loss='poisson',final_activation='sigmoid', opt = 'adadelta', activation = 'relu'):
    # Import the necessay 
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten#, LeakyReLU
    from keras.layers import Convolution2D, MaxPooling2D
    np.random.seed(2016) # For reproducibility:
    #from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
    

    # Other parameters that are not worth changing for our data.
    batch_size = 128
    nb_classes = 1
    # Input image dimensions, need to add the 1 for CNN.
    if len(X.shape)==4:
        N_samples,channels,img_rows,img_cols = X.shape
    elif len(X.shape)==3:
        N_samples,img_rows,img_cols = np.size(X,1), np.size(X,2)
        channels = 1
        X = X.reshape(N_samples,channels,img_rows,img_cols)
    #pix = img_rows*img_cols
    
    # Split the data:
    X_train, X_test = X[train_idxs,:,:,:], X[test_idxs,:,:,:]
    Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    
    # Clean up the data and convert for GPU:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(channels, img_rows, img_cols)))
    model.add(Activation(activation))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(nb_dense))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation(final_activation))
    
    model.compile(loss=loss,
                  optimizer= opt,
                  metrics=['accuracy'])
                  
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=2, validation_data=(X_test, Y_test),shuffle=True)
    import time
    time.sleep(1)
    if len(X_test)>0:
        score =[]
        Y_pred = model.predict(X_test);
        dc = dice_coef(Y_test, Y_pred);
        score = dc
        print('Dice coefficient: ' + str(dc))
        
    #        #score = model.evaluate(X_test, Y_test, verbose=0)
    #        #print('Test score:', score[0])
    #        #print('Test accuracy:', score[1])
    #        
    #        y_pred = model.predict_classes(X_test)
    #        y_pred = y_pred[:,0]
    #        accuracy = accuracy_score(Y_test, y_pred)
    #        recall = recall_score(Y_test, y_pred)
    #        precision = precision_score(Y_test, y_pred)
    #        f1 = f1_score(Y_test, y_pred)
    #        print('Accuracy: {}'.format(accuracy))
    #        print('Recall: {}'.format(recall))
    #        print('Precision: {}'.format(precision))
    #        print('F1: {}'.format(f1))
    #        score = np.array([accuracy, recall, precision, f1])
    else:
        score = []
    return model, score





