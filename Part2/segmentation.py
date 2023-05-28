from PIL import Image, ImageOps
import numpy as np
import cv2

# read file and convert to grey scale
file = "Part2/Dataset/CarPlate/10.jpg"
img = Image.open(file).convert('L')
img = ImageOps.exif_transpose(img) # Make sure image remains in original orientation

# Image resize 
size = (900,  900)  
img =  img.resize(size)
img_arr = np.array(img) # Image array 
greyscale_img = img.resize(size) # Greyscale image

# otsu method
greyscale_img_arr = np.array(greyscale_img)
_, threshold_image = cv2.threshold(greyscale_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # output either 0 or 255

# vertical projection
vertical_projection = np.sum(threshold_image, axis=0)
vertical_projection = vertical_projection/255


####################### Plotting ####################### TODO: DELETE 

# height, width = img.size
# blankImage = np.zeros_like(greyscale_img)

# for i, value in enumerate(vertical_projection):
#     cv2.line(blankImage, (i, 0), (i, height-int(value)), (255, 255, 255), 1)
# # print((blankImage.T)[:, -1])
# # Naming a window
# cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
# # Using resizeWindow()
# cv2.resizeWindow("Resized_Window", 800, 500)
# # Displaying the image
# cv2.imshow("Resized_Window", blankImage)
# # cv2.imshow('s2s', blankImage)
# cv2.waitKey(0)

########################################################

# set a threshold
threshold = 0.12 * np.max(vertical_projection)

# detect gaps
gaps = np.where(vertical_projection < threshold)[0]

# initialise variables
segments = []
start = 0

# segment characters
for gap in gaps:  
    # check if there is character
    if gap - start > 1: 
        char_segment = threshold_image[:, start:gap]  # extract the character
        segments.append(char_segment)  # save it to array
    start = gap + 1


# display character 
for i, segment in enumerate(segments):
    if segment.size > 0:
        cv2.namedWindow(f"Character {i+1}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Character {i+1}", 400, 500)
        cv2.imshow(f"Character {i+1}", segment)

cv2.waitKey(0)
cv2.destroyAllWindows()

