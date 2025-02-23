
from PIL import Image, ImageOps
import numpy as np
from gaussian import gaussian
from unsharp import unsharp
from sobel import vertical_sobel 
from search_carplate import search_carplate
from Part1.remove_edge_noise import remove_edge_noise
import matplotlib.pyplot as plt
from segmentation import remove_background

"""
Main function to run the algorithm, which will invoke various methods to run the entire 
car plate detection algorithm.  
"""
def main():

    ###### Open image 
    # To do: Chnage input file here  
    file = "Images/Set1/013.jpg"
    img = Image.open(file).convert('L')
    img = ImageOps.exif_transpose(img) # Make sure image remains in original orientation

    ###### Image resize 
    size = (900,  900)  
    img =  img.resize(size)
    img_arr = np.array(img) # Image array 
    greyscale_img = img.resize(size) # Greyscale image

    ###### Unsharp image to remove blur & increase contrast with histogram equalisation 
    img = unsharp(3, img)
    img = ImageOps.equalize(img, mask = None)
    
    ###### Gaussian blurring to remove noise 
    img = gaussian(5, img)

    ##### Vertical edge detection 
    edge_img = vertical_sobel(img, img_arr)

    ##### Search carplate 
    window_size_arr = [(130, 335), (100, 275), (70, 220)]
    car_plate_position = None
    iter = 0 

    # If cannot find carplate with current window size, switch 
    while car_plate_position == None and iter < len(window_size_arr):
        edge_img = remove_edge_noise(edge_img, window_size_arr[iter])
        car_plate_position = search_carplate(window_size_arr[iter], edge_img, greyscale_img)
        iter += 1
    
    ###### If unable to locate car plate:
    if car_plate_position == None:
        print("Cannot find carplate!")
    ####  Car plate detected
    else:
        # Crop our poistion and show
        result_img = np.array(img)
        height1 = car_plate_position[0]
        width1 = car_plate_position[1]
        result_img = result_img[(height1):(height1+window_size_arr[iter-1][0])]
        result_img = result_img[:, (width1):(width1+window_size_arr[iter-1][1])]
        result_img = Image.fromarray(result_img)
        result_img.show() 


if __name__ == "__main__": 
    main()
