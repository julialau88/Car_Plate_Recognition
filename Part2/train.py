from ann import read_files, Weight_Bias_Correction_Hidden, Weight_Bias_Correction_Output, Weight_Bias_Update,Weight_Initialization, Saving_Weights_Bias, Forward_Hidden_Output, Forward_Input_Hidden, Check_for_End
from sobel import sobel
import numpy as np
import cv2
from gaussian import gaussian
from PIL import Image
from unsharp import unsharp

###### Main function to train the ann
def main():
    ### Initialisation of parameters 
    alphabet = 'B'
    file = "Part2/Dataset/Alphabets/" + alphabet + "/1.jpg"

    # Input arr is the image
    input_img, input_img_arr, width, height = read_files(file)

    # Input_Neurons: Initialise total number of neurons
    Input_Neurons = width*height

    # Hidden_Neurons: Initialise number of hidden neurons
    Hidden_Neurons = 15

    # Target here is the expected output 
    # target_values = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    target_values = ["B", "F"]
    target_arr = np.zeros(len(target_values))

    index = target_values.index(alphabet)
    target_arr[index] = 1
    # Output_Neurons: Initialise number of output neurons
    Output_Neurons = len(target_values)

    # Apply Canny edge detection to the image
    input_arr = gaussian(5, input_img)

    # Image.fromarray(input_arr).show()
    input_arr = cv2.Canny(np.array(input_img_arr), 100, 200)

    _, threshold_image = cv2.threshold(input_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Image.fromarray(threshold_image).show()
    input_arr = np.divide(threshold_image, 255)

    # input_arr = sobel(width, height, input_arr)
    input_arr = input_arr.flatten()



    # Weight initialisation
    # If value has been saved before, use those values 
    # Else, initialise 
    try: 
        wji = np.load("wji.npy")
        wkj = np.load("wkj.npy")
        bias_j = np.load("bias_j.npy")
        bias_k = np.load("bias_k.npy")
    except FileNotFoundError:
        wji, wkj, bias_j, bias_k = Weight_Initialization(Input_Neurons, Hidden_Neurons, Output_Neurons)
    iter = 0
    max_iter = 30 
    error_threshold = 0.01

    target_index = 0 

    while True: 
        print("=== ITER:",  iter)
        NetJ = np.zeros(Hidden_Neurons)
        OutJ = np.zeros(Hidden_Neurons)
        NetJ, OutJ = Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_arr, bias_j, NetJ, OutJ, wji)

        NetK = np.zeros(Output_Neurons)
        OutK = np.zeros(Output_Neurons)
        NetK, OutK = Forward_Hidden_Output(wkj, Output_Neurons, Hidden_Neurons, OutJ, bias_k, NetK, OutK)

        print("THE RESULT IS:", OutK)
        # break
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

        if target_index >= len(target_arr):
            target_index = 0 
        else:
            target_index += 1
main()