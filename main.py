# testing pycharm git --ziqian
from sobel import vertical_sobel 
from PIL import Image
import numpy as np
from gaussian import gaussian
from unsharp import unsharp

def main():
    img = Image.open("Images/test_img4.png").convert('L')
    size = (1440, 1440)     # Resize to 1080 by 1080 
    img =  img.resize(size)
    img_arr = np.array(img)
    img = gaussian(5, img)

    img = unsharp(3, img)
    # print(img_arr)

    # width, height = img.size

    vertical_sobel(img, img_arr)

if __name__ == "__main__": 
    main()
