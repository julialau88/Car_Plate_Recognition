from PIL import Image, ImageOps
import numpy as np
import cv2
from ann import read_files, Weight_Bias_Correction_Hidden, Weight_Bias_Correction_Output, Weight_Bias_Update,Weight_Initialization, Saving_Weights_Bias, Forward_Hidden_Output, Forward_Input_Hidden, Check_for_End
from sobel import sobel
import numpy as np
import cv2
from gaussian import gaussian
from PIL import Image
from unsharp import unsharp

# read file and convert to grey scale
file = "Part2/Dataset/CarPlate/7.jpg"
img = Image.open(file).convert('L')
img = ImageOps.exif_transpose(img) # Make sure image remains in original orientation

# Image resize 
size = (900,  900)  
img =  img.resize(size)
img_arr = np.array(img) # Image array 
greyscale_img = img.resize(size) # Greyscale image
target_values = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W", "0","1", "2", "3", "4", "5", "6", "7", "8", "9"]

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
        char_segment = greyscale_img_arr[:, start:gap]  # extract the character
        segments.append(char_segment)  # save it to array
    start = gap + 1


# # display character 
# for i, segment in enumerate(segments):
#     if segment.size > 0:
#         cv2.namedWindow(f"Character {i+1}", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(f"Character {i+1}", 400, 500)
#         cv2.imshow(f"Character {i+1}", segment)

# display character 
string = ""
for i, segment in enumerate(segments):
    if segment.size > 0:
        # cv2.namedWindow(f"Character {i+1}", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(f"Character {i+1}", 400, 500)
        # cv2.imshow(f"Character {i+1}", segment)
        input_img = Image.fromarray(segment)
        # # ###### Image resize 
        size = (30,  20)  
        input_img =  input_img.resize(size)
        # input_img.show()

        input_img_arr = np.array(input_img) 
        width, height = input_img.size

        # exit()

        # Input_Neurons: Initialise total number of neurons
        Input_Neurons = width*height

        # Hidden_Neurons: Initialise number of hidden neurons
        Hidden_Neurons = 30

        # Output_Neurons: Initialise number of output neurons
        Output_Neurons = len(target_values)

        # Apply Canny edge detection to the image
        input_arr = gaussian(5, input_img)
        input_arr = sobel(width, height, input_arr)

        x_max = np.max(input_arr)
        x_min = np.min(input_arr)

        ## Normalise
        for row in range(0, height):
            for column in range(0, width):
                input_arr[row][column] = (((input_arr[row][column] - x_min)/(x_max- x_min))*2) - 1

        input_arr = input_arr.flatten()

        wji = np.load("wji.npy")
        wkj = np.load("wkj.npy")
        bias_j = np.load("bias_j.npy")
        bias_k = np.load("bias_k.npy")

        # print(input_arr)
        NetJ = np.zeros(Hidden_Neurons)
        OutJ = np.zeros(Hidden_Neurons)
        NetJ, OutJ = Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_arr, bias_j, NetJ, OutJ, wji)

        NetK = np.zeros(Output_Neurons)
        OutK = np.zeros(Output_Neurons)
        NetK, OutK = Forward_Hidden_Output(wkj, Output_Neurons, Hidden_Neurons, OutJ, bias_k, NetK, OutK)

        max_index = np.argmax(OutK)
        string += target_values[max_index]

print("THE RECOGNISED CARPLATE IS:", string)


cv2.waitKey(0)
cv2.destroyAllWindows()

