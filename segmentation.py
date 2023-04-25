from PIL import Image, ImageChops, ImageFilter, ImageOps
import numpy as np
import cv2

def remove_background(image):
    # Load the image
    # image = Image.open("Images/Set1/010.jpg")
    # image = ImageOps.exif_transpose(image)

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
    # segmented_image.show()
    # Image.fromarray(segmented_image).show()

    return segmented_image

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                        (img_height - crop_height) // 2,
                        (img_width + crop_width) // 2,
                        (img_height + crop_height) // 2))

def cropping(width, height):
    if width > height:
        width_new = width // 2
        return width_new
    else:
        height_new = height //2
        return height_new