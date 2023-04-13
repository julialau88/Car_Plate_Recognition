import numpy as np
from scipy.signal import convolve2d
from PIL import Image

def vertical_sobel(img, img_arr):
    # Kernel in y direction 
    Kx = np.matrix("-1, -2, -1; 0, 0, 0;, 1, 2, 1")
    Ky = np.matrix("-1, 0, 1; -2, 0, 2;, -1, 0, 1")
    
    # Apply sobel 
    Gx = convolve2d(img_arr, Kx, "same")
    Gy = convolve2d(img_arr, Ky, "same")

    # Threshold 
    T = np.mean(Gy) * 4
    
    # Use this threshold and apply nonmaximum suppression
    # in horizontal direction in the gradient
    # image, and we get the vertical Sobel edge image
    # shown in Fig. 6.

    # Finding Magnitude
    M = np.sqrt(Gy**2 + Gx**2)
    
    Gy = np.clip(Gy, 0, 255)
    Gx = np.clip(Gx, 0, 255)
    print(Gy)
    


