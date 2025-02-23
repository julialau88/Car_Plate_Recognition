from PIL import ImageFilter, ImageChops

"""
Function to sharpen image 
@param kernel_size: the size of the kernel 
@param img: the image to be sharpened
@return: sharpened image 
"""
def unsharp(kernel_size, img):

    # Blur with median flitering 
    img_blur = img.filter(ImageFilter.MedianFilter((kernel_size)))

    # Subtract to form unsharp mask
    unsharp_mask = ImageChops.subtract(img, img_blur)

    # Add to form sharpened image 
    sharped_img = ImageChops.add(img, unsharp_mask)

    return sharped_img
