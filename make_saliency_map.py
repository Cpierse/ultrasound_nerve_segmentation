# -*- coding: utf-8 -*-
"""
Created on Sat Jul 09 23:55:50 2016
Make saliency maps
@author: Chris Pierse
"""
import sys
import numpy as np
# Append location of saliency map code:
sys.path.append("saliency-map-master")
from saliency_map import SaliencyMap


def saliency_map(img):
    img3 = np.zeros([img.shape[0],img.shape[1],3])
    for i in range(3):
        img3[:,:,i] = img
    sm = SaliencyMap(img3)
    return sm.map

def saliency_maps(imgs):
    sms = np.zeros(imgs.shape)
    for i in range(imgs.shape[0]):
        sms[i,0,:,:] = saliency_map(imgs[i,0,:,:])
    return sms
