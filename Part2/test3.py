from ann import read_files, Weight_Bias_Correction_Hidden, Weight_Bias_Correction_Output, Weight_Bias_Update,Weight_Initialization, Saving_Weights_Bias, Forward_Hidden_Output, Forward_Input_Hidden, Check_for_End
from sobel import sobel
import numpy as np
import cv2
from gaussian import gaussian
from PIL import Image
from unsharp import unsharp

iter = 0
max_iter = 100
error_threshold = 0.0000001
target_index = 0    # Train which letter
file_index = 1      # Using which file 
target_values = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W", "0","1", "2", "3", "4", "5", "6", "7", "8", "9"]

for j in range(0, len(target_values)):
    target_arr = np.zeros(len(target_values))

    alphabet = target_values[j]
    files = [9,10]
    file_index = 10
    index = target_values.index(alphabet)
    target_arr[index] = 1

    # Based on alphabet, set 1 
    target_arr = np.zeros(len(target_values))
    index = target_values.index(alphabet)
    target_arr[index] = 1

    for i in range(0, len(files)):
        file = "Part2/Dataset/Alphabets/" + alphabet + "/" + str(files[i]) + ".jpg"
        # Input arr is the image
        input_img, input_img_arr, width, height = read_files(file)

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

        # Image.fromarray(input_arr).show()
        # input_arr = sobel(width, height, input_arr)
        # print(input_arr)
        ## Normalization 
        x_max = np.max(input_arr)
        x_min = np.min(input_arr)

        for row in range(0, height):
            for column in range(0, width):
                input_arr[row][column] = (((input_arr[row][column] - x_min)/(x_max- x_min))*2) - 1

        input_arr = input_arr.flatten()


        wji = np.load("wji.npy")
        wkj = np.load("wkj.npy")
        bias_j = np.load("bias_j.npy")
        bias_k = np.load("bias_k.npy")


        print("=== ITER:",  iter)
        print("TESTING ALPHABET", alphabet)
        print("TARGET ARR IS", target_arr)
        print("FILE IS", file)
        # print(input_arr)
        NetJ = np.zeros(Hidden_Neurons)
        OutJ = np.zeros(Hidden_Neurons)
        NetJ, OutJ = Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_arr, bias_j, NetJ, OutJ, wji)

        NetK = np.zeros(Output_Neurons)
        OutK = np.zeros(Output_Neurons)
        NetK, OutK = Forward_Hidden_Output(wkj, Output_Neurons, Hidden_Neurons, OutJ, bias_k, NetK, OutK)

        print("THE RESULT IS:", OutK)
        max_index = np.argmax(OutK)
        print("THE RECOGNISED CHARACTER IS:", target_values[max_index])
