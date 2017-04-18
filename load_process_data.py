# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 22:35:24 2016
Loads and processes the data. Can also prep submission.

@author: Chris Pierse
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


#%% Key functions:
def find_imgs(file_dir):
    files = os.listdir(file_dir)
    imgs = [x for x in files if ('mask.tif' not in x) and ('.tif' in x)]
    try: 
        imgs = sorted(imgs, key=lambda x: float(x[0:len(x)-4]))
    except: 
        lefts = np.unique([str.split(x,'_')[0] for x in imgs ])
        lefts = sorted(lefts,key=float)
        n_imgs = []
        for i in lefts:
            idx = 1
            while True:
                string = i + '_' + str(idx)+'.tif' 
                if string in imgs:
                    n_imgs.append(string)
                    idx+=1
                else:
                    break
        imgs = n_imgs
    masks = [x[0:len(x)-4]+'_mask.tif' for x in imgs ]
    names =  [x[0:len(x)-4] for x in imgs]
    save_txt(os.path.join(file_dir,'names.txt'),names)
    return imgs,masks, names

def save_txt(name,txt_list):
    f=open(name,'w')
    for ele in txt_list:
        f.write(ele+'\n')
    f.close()

def load_convert_imgs(paths,s_rows,s_cols):
    X = np.zeros([len(paths),1,s_rows,s_cols],'uint8')
    for i,path in enumerate(paths):
        img = cv2.imread(path,0)
        res = cv2.resize(img, (s_cols,s_rows), interpolation = cv2.INTER_AREA)
        X[i,:,:] = np.array(res,'uint8')
    return X

def load_data(s_rows = 80, s_cols = 112, file_dir = 'train'):
    imgs,masks,names = find_imgs(file_dir)
    imgs = load_convert_imgs([file_dir + '/' + x for x in imgs],s_rows,s_cols)
    if file_dir == 'test': masks = []
    else: 
        masks = load_convert_imgs([file_dir + '/' + x for x in masks],s_rows,s_cols)
        masks[masks>0.1] = 1
        masks[masks<=0.1] = 0
    return imgs, masks, names

def load_matlab_csv(s_rows = 80, s_cols = 112):#, file_dir = 'train'):
    imgs = np.genfromtxt( 'Full_Imgs_' + str(s_rows) + 'x' + str(s_cols) + '.csv', delimiter=',')
    masks = np.genfromtxt( 'Full_Masks_' + str(s_rows) + 'x' + str(s_cols) + '.csv', delimiter=',')
    # Reshape the array:
    imgs = imgs.reshape(imgs.shape[0],1,s_cols,s_rows).transpose(0,1,3,2) # Note: rotated compared to Matlab.
    masks = masks.reshape(masks.shape[0],1,s_cols,s_rows).transpose(0,1,3,2) # Note: rotated compared to Matlab.
    plt.imshow(imgs[0,0,:,:],cmap='gray')
    return imgs,masks

def convert_to_csv(imgs,name):
    if len(imgs.shape)==3:
        N,rows,cols = imgs.shape
        chans = 1
    elif len(imgs.shape)==4:
        N,chans,rows,cols = imgs.shape
    data = np.zeros([N,rows*cols])
    for i in range(chans):
        for j in range(N):
            data[j,:] = imgs[j,i,:,:].flatten()
        txt_name = '_'.join([name,str(i),str(rows)+'x'+str(cols)])+'.csv'
        np.savetxt(txt_name, data, delimiter=',', newline='\n')


def make_submission(Y_pred,names,o_rows=420, o_cols=580):
    N = Y_pred.shape[0]
    Y_scaled = np.zeros([N,o_rows,o_cols])
    for i in range(Y_scaled.shape[0]):
        Y_scaled[i,:,:] = cv2.resize(Y_pred[i,0,:,:], (o_cols,o_rows), interpolation = cv2.INTER_CUBIC)
    # Record rles
    rles = []
    for i in range(N):
        rle = run_length_enc(Y_scaled[i,:,:])
        rles.append(rle)
    # Prepare file intro and name
    first_row = 'img,pixels'
    file_name = 'submission.csv'
    # Write to file
    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(N):
            s = str(names[i]) + ',' + rles[i]
            f.write(s + '\n')
    print('Submission Ready')
    

def run_length_enc(mask):
    # Borrowed from another user on Kaggle:
    from itertools import chain
    x = mask.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])






#%% Scratch:








