from PIL import Image, ImageChops
import numpy as np
import cv2

"""
Background removal function
@param image: image to remove background from 
@return image with background removed
"""
def remove_background(image):

    # Convert the image to grayscale
    gray_image = image.convert("L")

    # Apply Canny edge detection to the image
    canny_image = cv2.Canny(np.array(gray_image), 100, 200)

    # Create a mask with the car in white and everything else in black
    mask = Image.fromarray(canny_image).convert("1")

    # Apply the mask to the original image
    segmented_image = ImageChops.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)
    width_segmented, height_segmented = segmented_image.size
    segmented_image = crop_center(image, cropping(width_segmented, height_segmented), cropping(width_segmented, height_segmented))

    return segmented_image

"""
This function is used to set the 9 quadrant in the image based on the input image. After setting the quadrant,
The function will find the quadrant with the most black pixels among the 9 quadrant. Moreover, the function will find among
the centre quadrant to find which has the most black pixels. Furthermore, after getting the coordinate of the area
of image that needed to be cropped then it will used .crop function to crop the image into the desired dimension.
@param pil_img: image
@param crop_width: the crop width
@param crop_height: the crop height 
"""
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                        (img_height - crop_height) // 2,
                        (img_width + crop_width) // 2,
                        (img_height + crop_height) // 2))

"""
This function is to decide the input width and the input height for the crop_center function as to get the desired 
dimensiion of the output.
@param width: width of image 
@param height: height of image
@return new height or new width
"""
def cropping(width, height):
    if width > height:
        width_new = width // 2
        return width_new
    else:
        height_new = height //2
        return height_new