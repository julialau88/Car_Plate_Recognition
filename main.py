
from PIL import Image, ImageOps
import numpy as np
from gaussian import gaussian
from unsharp import unsharp
from sobel import vertical_sobel 
from search_carplate import search_carplate
from remove_edge_noise import remove_edge_noise
from skimage import filters
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def main():
    img = Image.open("Images/Set1/001.jpg").convert('L')
    img = ImageOps.exif_transpose(img)

    size = (900,  900)     # Resize to 1080 by 1080 
    img =  img.resize(size)
    greyscale_img = img.resize(size)
    img_arr = np.array(img)

    # Show image on plt
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plt.rc('image', cmap='gray')
    ax.set_title('Original Image'), ax.imshow(img)
    plt.show()

    img = gaussian(5, img)

    img = unsharp(3, img)

    edge_img = vertical_sobel(img, img_arr)

    edge_img = remove_edge_noise(edge_img)
    edge_img.show()

    # Next, search carplate 
    # (height, width)
    # window_size = (150, 330) # window size for close up image 
    window_size = (75, 220) 

    pixel_range = (window_size[0]*window_size[1]*0.21)
    search_carplate(pixel_range, window_size, edge_img, greyscale_img)


    # edge_img1 = np.array(edge_img)
    # height1 = 555
    # width1 = 240
    # edge_img1 = edge_img1[height1:height1+window_size[0]]
    # edge_img1 = edge_img1[:, width1:(width1+window_size[1])]
    # kernel = np.ones((4,10))
    # G = convolve2d(edge_img1, kernel, "same")
    # # print(edge_img1)
    # edge_img1 = Image.fromarray(edge_img1)
    # edge_img1.show() 


if __name__ == "__main__": 
    main()
