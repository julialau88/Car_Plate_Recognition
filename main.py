
from sobel import vertical_sobel 
from PIL import Image
import numpy as np

def main():
    img = Image.open("Images/test_img1.png").convert('L')
    size = (384, 288)
    img =  img.resize(size)
    img_arr = np.array(img)
    print(img_arr)
    # width, height = img.size

    vertical_sobel(img, img_arr)

if __name__ == "__main__": 
    main()