import numpy as np
from PIL import ImageFilter

"""
Gaussian smoothing
@param window_size: size of Gaussian mask 
@param img: image to apply Gaussian on 
@return Gaussian smoothed image
"""
def gaussian(window_size, img):
    # Window size 
    window = window_size**2

    # Kernel 
    k = np.ones(window)/window

    # Gaussian 
    img_filter_gaussian = img.filter(ImageFilter.Kernel((window_size,window_size), k))

    return img_filter_gaussian