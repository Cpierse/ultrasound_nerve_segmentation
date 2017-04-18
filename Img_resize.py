# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 23:01:18 2016

@author: Chris Pierse
"""

def resize(image,scale):
    from PIL import Image
    import numpy as np
    scale = float(scale)
    new_width = np.int(np.round(scale*image.shape[1]))
    new_height = np.int(np.round(scale*image.shape[0]))

    img = Image.fromarray(image)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    image = np.array(img)
    return image