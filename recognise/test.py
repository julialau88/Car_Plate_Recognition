"""
File to test on training characters
"""
from ann import read_files, Forward_Hidden_Output, Forward_Input_Hidden
import numpy as np
import cv2

target_index = 0    # Train which letter
file_index = 1      # Using which file 
target_values = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W", "0","1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Loop through all characters 
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
        file = "Part2/Dataset/Characters/" + alphabet + "/" + str(files[i]) + ".jpg"
        # Input arr is the image
        input_img, input_img_arr, width, height = read_files(file)

        # Input_Neurons: Initialise total number of neurons
        Input_Neurons = width*height

        # Hidden_Neurons: Initialise number of hidden neurons
        Hidden_Neurons = 100

        # Output_Neurons: Initialise number of output neurons
        Output_Neurons = len(target_values)

        # Apply Otsu's to the image
        _, threshold_image = cv2.threshold(input_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        input_arr = np.divide(threshold_image, 255)
        input_arr = input_arr.flatten()

        wji = np.load("Part2/Weights/wji.npy")
        wkj = np.load("Part2/Weights/wkj.npy")
        bias_j = np.load("Part2/Weights/bias_j.npy")
        bias_k = np.load("Part2/Weights/bias_k.npy")

        # Forward propagation 
        NetJ = np.zeros(Hidden_Neurons)
        OutJ = np.zeros(Hidden_Neurons)
        NetJ, OutJ = Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_arr, bias_j, NetJ, OutJ, wji)

        NetK = np.zeros(Output_Neurons)
        OutK = np.zeros(Output_Neurons)
        NetK, OutK = Forward_Hidden_Output(wkj, Output_Neurons, Hidden_Neurons, OutJ, bias_k, NetK, OutK)

        # Print output
        print("TESTING ALPHABET", alphabet)
        print("TARGET ARR IS", target_arr)
        print("FILE IS", file)
        print("THE RESULT IS:", OutK)
        max_index = np.argmax(OutK)
        print("THE RECOGNISED CHARACTER IS:", target_values[max_index])
