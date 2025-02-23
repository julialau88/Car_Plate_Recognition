"""
Train ANN model
"""
from ann import read_files, Weight_Bias_Correction_Hidden, Weight_Bias_Correction_Output, Weight_Bias_Update,Weight_Initialization, Saving_Weights_Bias, Forward_Hidden_Output, Forward_Input_Hidden, Check_for_End
import numpy as np
import cv2

iter = 0
max_iter = 20000
error_threshold = 0.0003
target_index = 0    # Train which letter
file_index = 1      # Using which file 
read = False

while True: 
    target_values = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    alphabet = target_values[target_index]
    file = "Part2/Dataset/Characters/" + alphabet + "/" + str(file_index) + ".jpg"

    # Based on alphabet, set 1 in target arr
    target_arr = np.zeros(len(target_values))
    index = target_values.index(alphabet)
    target_arr[index] = 1

     # Input arr is the image
    input_img, input_img_arr, width, height = read_files(file)

    # Input_Neurons: Initialise total number of input neurons, hidden neurons, and output neurons
    Input_Neurons = width*height
    Hidden_Neurons = 100
    Output_Neurons = len(target_values)

    # Apply Otsu's to the image
    _, threshold_image = cv2.threshold(input_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    input_arr = np.divide(threshold_image, 255)
    input_arr = input_arr.flatten()

    if not read:
        wji, wkj, bias_j, bias_k = Weight_Initialization(Input_Neurons, Hidden_Neurons, Output_Neurons)
        read = True

    # Forward propagation
    NetJ = np.zeros(Hidden_Neurons)
    OutJ = np.zeros(Hidden_Neurons)
    NetJ, OutJ = Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_arr, bias_j, NetJ, OutJ, wji)

    NetK = np.zeros(Output_Neurons)
    OutK = np.zeros(Output_Neurons)
    NetK, OutK = Forward_Hidden_Output(wkj, Output_Neurons, Hidden_Neurons, OutJ, bias_k, NetK, OutK)


    print("=== ITER:",  iter)
    print("TRAINING ALPHABET", alphabet)
    print("TARGET ARR IS", target_arr)
    print("FILE IS", file)
    print("THE RESULT IS:", OutK)

    is_end, error_arr = Check_for_End(OutK, target_arr, iter, max_iter, error_threshold)
    iter += 1

    if is_end: 
        # Go to Step 10
        # Save weights and biases
        Saving_Weights_Bias(wji, bias_j, wkj, bias_k)
        break
    else: 
        delta_bias_k, delta_wk = Weight_Bias_Correction_Output(OutK, target_arr, OutJ, Hidden_Neurons, Output_Neurons)
        delta_WJ, delta_bias_j  = Weight_Bias_Correction_Hidden(OutK, target_arr, OutJ, Hidden_Neurons, Output_Neurons, Input_Neurons, input_arr, wkj)
        wji_new, bias_j_new, wkj_new, bias_k_new   = Weight_Bias_Update(wji, wkj, Hidden_Neurons, Input_Neurons, Output_Neurons, delta_WJ, bias_j, delta_bias_j, delta_wk, bias_k, delta_bias_k)

        # Reassign wji, bias_j, wkj, bias_k
        bias_j_new = [[item] for item in bias_j_new]
        bias_k_new = [[item] for item in bias_k_new]
        wji, bias_j, wkj, bias_k = wji_new, bias_j_new, wkj_new, bias_k_new 

    if target_index >= len(target_arr)-1:
        target_index = 0 
        if file_index >= 8:
            file_index = 1
        else:
            file_index += 1
    else:
        target_index += 1
