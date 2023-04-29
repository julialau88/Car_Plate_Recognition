
from PIL import Image, ImageOps
import numpy as np
from gaussian import gaussian
from unsharp import unsharp
from sobel import vertical_sobel 
from search_carplate import search_carplate
from remove_edge_noise import remove_edge_noise
import matplotlib.pyplot as plt
from segmentation import remove_background

"""
Main function to run the algorithm, which will invoke various methods to run the entire 
car plate detection algorithm 
"""
def main():
    ###### Open image 
    # To do: Chnage input file here  
    file = "Images/Set1/013.jpg"
    img = Image.open(file).convert('L')
    img = ImageOps.exif_transpose(img) # Make sure image remains in original orientation
    # img = remove_background(img)
    
    ###### Image resize 
    size = (900,  900)  
    img =  img.resize(size)
    img_arr = np.array(img) # Image array 
    greyscale_img = img.resize(size) # Greyscale image

    img.save("greyscale.png")

    ###### Show image on plt
    # fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    # plt.rc('image', cmap='gray')
    # ax.set_title('Original Image'), ax.imshow(img)
    # plt.show()

    ###### Unsharp image to remove blur & increase contrast with histogram equalisation 
    img = unsharp(3, img)
    img.save("unsharp.png")
    img = ImageOps.equalize(img, mask = None)
    img.save("equalised.png")
    
    ###### Gaussian blurring to remove noise 
    img = gaussian(5, img)
    img.save("gaussian.png")
    # img.show() 

    ##### Vertical edge detection 
    edge_img = vertical_sobel(img, img_arr)
    edge_img.convert('RGB').save("edge.png")

    ##### Search carplate 
    window_size_arr = [(130, 335), (100, 275), (70, 220)]
    car_plate_position = None
    iter = 0 

    # If cannot find carplate with current window size, switch 
    while car_plate_position == None and iter < len(window_size_arr):
        edge_img = remove_edge_noise(edge_img, window_size_arr[iter])
        edge_img.convert('RGB').save("edge_noise_removed.png")
        car_plate_position = search_carplate(window_size_arr[iter], edge_img, greyscale_img)
        iter += 1
    
    ###### If unable to locate car plate:
    if car_plate_position == None:
        print("Cannot find carplate!")
    ####  Car plate detected
    else:
        # Crop our poistion and show
        edge_img1 = np.array(img)
        height1 = car_plate_position[0]
        width1 = car_plate_position[1]
        edge_img1 = edge_img1[(height1):(height1+window_size_arr[iter-1][0])]
        edge_img1 = edge_img1[:, (width1):(width1+window_size_arr[iter-1][1])]
        edge_img1 = Image.fromarray(edge_img1)
        edge_img1.save("result.png")
        edge_img1.show() 


if __name__ == "__main__": 
    main()
