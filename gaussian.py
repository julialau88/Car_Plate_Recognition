import numpy as np
from PIL import Image, ImageFilter

def gaussian(window_size, img):
    window = window_size**2
    print(window)
    k = np.ones(window)/window
    img_filter_gaussian = img.filter(ImageFilter.Kernel((window_size,window_size), k))

    return img_filter_gaussian