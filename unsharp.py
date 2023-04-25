import numpy as np
from PIL import Image, ImageFilter, ImageChops

def unsharp(kernel_size, img):
    kernel = kernel_size**2

    k = np.ones(kernel)/kernel

    # img_blur = img.filter(ImageFilter.Kernel((kernel_size, kernel_size), k))
    img_blur = img.filter(ImageFilter.MedianFilter((kernel_size)))

    unsharp_mask = ImageChops.subtract(img, img_blur)

    sharped_img = ImageChops.add(img, unsharp_mask)

    return sharped_img
