from ann import read_files, Weight_Bias_Correction_Hidden, Weight_Bias_Correction_Output, Weight_Bias_Update,Weight_Initialization, Saving_Weights_Bias, Forward_Hidden_Output, Forward_Input_Hidden, Check_for_End
from sobel import sobel
import numpy as np
import cv2
from gaussian import gaussian
from PIL import Image
from unsharp import unsharp
target_values = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W", "0","1", "2", "3", "4", "5", "6", "7", "8", "9"]
alphabet = "B"
for i in range(1, 11):
    file = "Part2/Dataset/Alphabets/" + alphabet + "/" + str(i) + ".jpg"
    # Input arr is the image
    input_img, input_img_arr, width, height = read_files(file)


    # exit()

    # Input_Neurons: Initialise total number of neurons
    Input_Neurons = width*height

    # Hidden_Neurons: Initialise number of hidden neurons
    Hidden_Neurons = 300

    # Output_Neurons: Initialise number of output neurons
    Output_Neurons = len(target_values)

    # Apply Canny edge detection to the image
    input_arr = gaussian(5, input_img)

    # Image.fromarray(input_arr).show()
    input_arr = cv2.Canny(np.array(input_img_arr), 100, 200)
    _, threshold_image = cv2.threshold(input_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    Image.fromarray(threshold_image).show()
