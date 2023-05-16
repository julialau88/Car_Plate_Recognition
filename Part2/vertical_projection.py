from PIL import Image, ImageOps
import numpy as np
import cv2


file = "Part2/Dataset/CarPlate/1.jpg"
img = Image.open(file).convert('L')
img = ImageOps.exif_transpose(img) # Make sure image remains in original orientation

###### Image resize 
size = (900,  900)  
img =  img.resize(size)
img_arr = np.array(img) # Image array 
greyscale_img = img.resize(size) # Greyscale image


greyscale_img_arr = np.array(greyscale_img)
_, threshold_image = cv2.threshold(greyscale_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # output either 0 or 2555

vertical_projection = np.sum(threshold_image, axis=0)
vertical_projection = vertical_projection/255


height, width = img.size
blankImage = np.zeros_like(greyscale_img)

####################### Plotting #######################

for i, value in enumerate(vertical_projection):
    cv2.line(blankImage, (i, 0), (i, height-int(value)), (255, 255, 255), 1)
# print((blankImage.T)[:, -1])
# Naming a window
cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
# Using resizeWindow()
cv2.resizeWindow("Resized_Window", 800, 500)
# Displaying the image
cv2.imshow("Resized_Window", blankImage)
# cv2.imshow('s2s', blankImage)
cv2.waitKey(0)

########################################################
