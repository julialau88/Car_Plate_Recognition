
from PIL import Image, ImageOps
import numpy as np
from gaussian import gaussian
from unsharp import unsharp
from sobel import vertical_sobel 
from search_carplate import search_carplate
from remove_edge_noise import remove_edge_noise
from segmentation import remove_background
import matplotlib.pyplot as plt

"""
Main function to run the algorithm, which will invoke various methods to run the entire 
car plate detection algorithm 
"""
def main():
    for i in range(21, 45, 1):
        print("---------------------" + str(i) + "---------------------")
        temp = '0' + str(i) + '.jpg'
        file = "Images/Set1/" + temp
        img = Image.open(file).convert('L')
        img = ImageOps.exif_transpose(img)

        # Remove unnecessary background
        # img = remove_background(img)

        # Image resize 
        size = (900,  900)  
        img =  img.resize(size)
        img_arr = np.array(img) # Image array 
        greyscale_img = img.resize(size) # Greyscale image

        # Show image on plt
        # fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        # plt.rc('image', cmap='gray')
        # ax.set_title('Original Image'), ax.imshow(img)
        # plt.show()

        # Unsharp image to remove blur & increase contrast with histogram equalisation 
        img = unsharp(3, img)
        img = ImageOps.equalize(img, mask = None)

        # Gaussian blurring
        img = gaussian(5, img)

        # img.show() 

        edge_img = vertical_sobel(img, img_arr)

        # Next, search carplate 
        noise_removal_window_size = (130, 300)

        edge_img = remove_edge_noise(edge_img, noise_removal_window_size)
        edge_img.show()

        window_size, car_plate_position = search_carplate(edge_img, greyscale_img)
        # window_size_arr = [(130, 330), (100, 275), (70, 220)]

        if car_plate_position == None:
            print("Cannot find carplate!")
        else:
            edge_img1 = np.array(img)
            height1 = car_plate_position[0]
            width1 = car_plate_position[1]
            edge_img1 = edge_img1[(height1):(height1+window_size[0])]
            edge_img1 = edge_img1[:, (width1):(width1+window_size[1])]
            edge_img1 = Image.fromarray(edge_img1)
            edge_img1.show() 


if __name__ == "__main__": 
    main()
