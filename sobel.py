import numpy as np
from scipy.signal import convolve2d
from PIL import Image

"""
Non-maximum suppression to thin edges 
@param Gy: Vertical gradient array after Sobel 
@param height: height of image 
@param width: width of image 
@param T2: upper threshold 
@param T1: lower threshold

"""
def non_maximum_suppression(Gy, height, width, T2, T1):
    S = np.zeros((height,width))
    E = np.zeros((height,width))
    result = np.zeros((height,width))

   ## Non maximum suppression 
    for h in range(0, height):
        for w in range(0, width):  
            # Examine horizontal pair 
            ## Checking edges first
            if w == 0:
                # Check right pixel
                if Gy[h][w+1] < Gy[h][w]:
                    E[h][w] = Gy[h][w]
                else: 
                    E[h][w] = 0
            elif w == width - 1:
                # Check left pixel 
                if Gy[h][w-1] < Gy[h][w]:
                    E[h][w] = Gy[h][w]
                else: 
                    E[h][w] = 0
            else:
                # Check both sides 
                if Gy[h][w-1] < Gy[h][w] and Gy[h][w+1] < Gy[h][w]:
                    E[h][w] = Gy[h][w]
                else:
                    E[h][w] = 0
   
    # Thresholding
    for h in range(0, height):
        for w in range(0, width):
            if E[h][w] > T2: 
                result[h][w] = 255
    
    # Hysteris 
    for h in range(0, height):
        for w in range(0, width):
            if E[h][w] < T2 and E[h][w] > T1: 
                # Checking neighbouring edges 
                if w == 0:
                    if h == 0: 
                        #Cannot check h-1 and w-1
                        if result[h][w+1] == 1 or result[h+1][w] == 1 or result[h+1][w+1] == 1:
                            result[h][w] = 1
                    elif h == height - 1:
                        # Cannot check h+1 and w-1
                        if result[h][w+1] == 1 or result[h-1][w] == 1 or result[h-1][w+1] == 1:
                            result[h][w] = 1
                    else: 
                        # Check bottom
                        if result[h][w+1] == 1 or result[h-1][w] == 1 or result[h-1][w+1] == 1 or result[h+1][w] == 1 or result[h+1][w+1] == 1:
                            result[h][w] = 1

                elif w == width - 1:
                    if h == 0:
                        # Cannot check top and right
                        if result[h][w-1] == 1 or result[h+1][w] == 1 or result[h+1][w-1] == 1:
                            result[h][w] = 1
                    elif h == height - 1:
                        # Cannot check bottom and right
                        if result[h][w-1] == 1 or result[h-1][w] == 1 or result[h-1][w-1] == 1:
                                result[h][w] = 1 
                elif h == 0 and w != 0 and w != width - 1:
                    # Cannot check top
                    if result[h][w-1] == 1 or result[h+1][w] == 1 or result[h+1][w-1] == 1 or result[h][w+1] == 1 or result[h+1][w+1] == 1:
                        result[h][w] = 1 
                elif h == height - 1 and w != 0 and w != width - 1:
                    # Cannot check bottom 
                    if result[h][w+1] == 1 or result[h-1][w] == 1 or result[h-1][w+1] == 1 or result[h][w-1] == 1  or result[h-1][w-1] == 1:
                        result[h][w] = 1 
                else:
                    # Not edges, check all 
                    if result[h][w+1] == 1 or result[h-1][w] == 1 or result[h-1][w+1] == 1 or result[h][w-1] == 1  or result[h-1][w-1] == 1 or result[h+1][w] == 1 or result[h+1][w-1] == 1 or result[h+1][w+1] == 1:
                        result[h][w] = 1 
    # Return image 
    edge_img = Image.fromarray(result)

    return edge_img

"""
Vertical Sobel to map out vertical edges 
@param img: the greyscale enhanced image
@param img_arr: greyscale enchanced image array 
@return the edge img 
"""
def vertical_sobel(img, img_arr):
    # Kernel in y direction 
    Ky = np.matrix("-1, 0, 1; -2, 0, 2;, -1, 0, 1")
    width, height = img.size # Get image size 
    
    # Apply sobel in vertical direction
    Gy = convolve2d(img_arr, Ky, "same")

    # Thresholds - absolute mean gradient value multiplied by a coefficient
    T2 = np.mean(np.abs(Gy)) * 5 # Upper
    T1 = np.mean(np.abs(Gy)) * 4 # Lower

    # Do non-maximum suppression 
    return non_maximum_suppression(Gy, height, width, T2, T1)



