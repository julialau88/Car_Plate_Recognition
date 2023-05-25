import numpy as np 
from PIL import Image
from scipy.signal import convolve2d
import math
from sobel import sobel

##### Step 1: initialisation of weights 
def Weight_Initialization(Input_Neurons, Hidden_Neurons, Output_Neurons):
    # Initializing of the Weights.
    # Random float number between -0.5 to 0.5.
    wji= np.random.uniform(-0.5, 0.5, size=(Hidden_Neurons, Input_Neurons))
    wkj = np.random.uniform(-0.5, 0.5, size=(Output_Neurons, Hidden_Neurons))
    bias_j = np.random.uniform(0, 1, size=(Hidden_Neurons, 1))          # In this format: [[0.32650096][0.16459484][0.07902434][0.28043552][0.68215983]]
    bias_k = np.random.uniform(0, 1, size=(Output_Neurons, 1))

    return wji, wkj, bias_j, bias_k

##### Step 2. Reading of Training Files, and Target Files.
def read_files(file):
    # Reading of Input File, and Target File.
    # Ask the user to enter the number of input, hidden and output neurons.
    input_img = Image.open(file).convert('L')
    
    # # ###### Image resize 
    size = (40,  50)  
    input_img =  input_img.resize(size)

    input_img_arr = np.array(input_img) 
    width, height = input_img.size

    return input_img,  input_img_arr, width, height

##### Step 3. Forward Propagation from Input -> Hidden Layer.
def Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_img_arr, bias_j, NetJ, OutJ, wji):

    # Obtain the results at each neuron in the hidden layer
    # Initialise NetJ --> same size as hidden neurons 

    # print(bias_j)
    # Loop through hidden neurons 
    for j in range(0, Hidden_Neurons):
        # For each hidden neuron. loop through input neurons to calculate NetJ
        for i in range(0, Input_Neurons):
            # NetJ[j] += (wj * x0) 
            NetJ[j] += wji[j][i]*input_img_arr[i]
        # NetJ[j] += bias_j
        NetJ[j] += bias_j[j][0]

    # Calculate OutJ
    for j in range(0,Hidden_Neurons):
        OutJ[j] = 1/(1+ math.exp(-(NetJ[j])))

    return NetJ, OutJ

##### Step 4. Forward Propagation from Hidden -> Output Layer.
def Forward_Hidden_Output(wkj, Output_Neurons, Hidden_Neurons, OutJ, bias_k, NetK, OutK):
    # Forward Propagation from Hidden -> Output Layer

    #For each output neuron, loop through hidden neuron to calculate NetK
    for k in range(0, Output_Neurons):
        for j in range(0, Hidden_Neurons):
            # NetK += wk * OutJ
            NetK[k] += wkj[k][j]* OutJ[j]
        # NetK += bias_k 
        NetK[k] += bias_k[k][0]
    
    # Calculate OutK with NetK
    # print(NetK)
    for k in range(0,Output_Neurons):
        OutK[k] = 1/(1 + math.exp(-(NetK[k])))
    
    return NetK, OutK

##### Step 5. Check for the global error or number of iterations.
def Check_for_End(OutK, target_arr, iter, max_iter, error_threshold):
    # Check whether the total error is less than the error set by the user or the number of iterations is reached.
    # Calculate total error 
    total_error = 0 
    error_arr = []
    ## Here want to set margin of error
    ## For example: if target value is 1, if ouput is 0.9, we consider it as pass
    for i in range(0, len(target_arr)):
        # print(target_arr)
        # print(OutK)
        error = 0.5*((target_arr[i] - OutK[i])**2)
        total_error += error
        error_arr.append(error)
        

    print("TOTAL ERROR IS",  total_error)
    # returns true or false
    if total_error <= error_threshold:
        print("ERROR THRESHOLD MET")
        return True, error_arr 

    # If TRUE, proceed to Step 10
    if iter >= max_iter - 1:
        print("MAX ITER REACHED")
        return True, error_arr

    return False, error_arr 

##### Step 6. Correction of Weights and Bias between Hidden and Output Layer.
def Weight_Bias_Correction_Output(OutK, target_arr, OutJ, Hidden_Neurons, Output_Neurons):
    # Correction of Weights and Bias between Hidden and Output Layer.
    # Calculate 𝑑𝑤𝑘𝑘𝑗 and 𝑑𝑏𝑘𝑘𝑗  
    delta_wk = np.zeros((Output_Neurons, Hidden_Neurons))
    delta_bias_k = np.zeros(Output_Neurons)

    for k in range(0, len(OutK)):
       delta_k_value = (OutK[k] - target_arr[k])*(OutK[k]*(1-OutK[k]))
       delta_bias_k[k] = (delta_k_value) 
    
    for k in range(0, len(OutK)):
        for j in range(0, len(OutJ)): 
            delta_wk[k][j] = delta_bias_k[k]*OutJ[j]
    return delta_bias_k, delta_wk 

##### Step 7. Correction of Weights and Bias between Input and Hidden Layer.
def Weight_Bias_Correction_Hidden(OutK, target_arr, OutJ, Hidden_Neurons, Output_Neurons, Input_Neurons, input_img_arr, wkj):
    # Correction of Weights and Bias between Input and Hidden Layer.
    # Calculate 𝑑𝑤𝑗𝑗𝑖 and 𝑑𝑏𝑗𝑗𝑖

    delta_kl = np.zeros(Output_Neurons)
    delta_WJ = np.zeros((Hidden_Neurons, Input_Neurons))
    delta_bias_j = np.zeros(Hidden_Neurons)
    sum_k_l = np.zeros(Hidden_Neurons)
    
    for k in range(0, len(OutK)):
        delta_k_value = (OutK[k] - target_arr[k])*(OutK[k]*(1-OutK[k]))
        delta_kl[k] =  delta_k_value
    
    for j in range(0, len(OutJ)):
        for l in range(0, len(OutK)):
            sum_k_l[j] +=  delta_kl[l]* wkj[l][j] 

    for j in range(0, len(OutJ)):
        for i in range(0, len(input_img_arr)):
            delta_WJ[j][i] = input_img_arr[i] * (OutJ[j]*(1-OutJ[j])) * sum_k_l[j]   
            delta_bias_j[j] = (OutJ[j]*(1-OutJ[j])) *  sum_k_l[j]   
    
    return  delta_WJ, delta_bias_j  

##### Step 8. Update Weights and Bias.
def Weight_Bias_Update(wji, wkj, Hidden_Neurons, Input_Neurons, Output_Neurons, delta_WJ, bias_j, delta_bias_j, delta_wk, bias_k, delta_bias_k):
    # Update Weights and Bias.
    # Calculate 𝑤𝑘𝑘𝑗+ and 𝑏𝑘𝑘𝑗+
    # Calculate 𝑤𝑗𝑗𝑖+ and 𝑏𝑗𝑗𝑖+
    n = 0.5

    wji_new= np.zeros((Hidden_Neurons, Input_Neurons))
    wkj_new = np.zeros((Output_Neurons, Hidden_Neurons))
    bias_j_new = np.zeros(Hidden_Neurons)
    bias_k_new = np.zeros(Output_Neurons)
    
    # Loop through hidden neurons 
    for j in range(0, Hidden_Neurons):
        # For each hidden neuron. loop through input neurons to calculate NetJ
        for i in range(0, Input_Neurons):
            wji_new[j][i] =   wji[j][i] - (n*delta_WJ[j][i])  
    
    for j in range(0, Hidden_Neurons):       
        bias_j_new[j] = bias_j[j] - (n*delta_bias_j[j]) 

    # Loop through output neurons 
    for k in range(0, Output_Neurons):
        # For each output neuron. loop through input neurons to calculate NetJ
        for j in range(0, Hidden_Neurons):
            wkj_new[k][j] =   wkj[k][j] - (n*delta_wk[k][j])  
    
    for k in range(0, Output_Neurons):
            bias_k_new[k] =   bias_k[k] - (n*delta_bias_k[k])   

    return wji_new, bias_j_new, wkj_new, bias_k_new        

###### Step 10. Save the Weights and Bias.
def Saving_Weights_Bias(wji, bias_j, wkj, bias_k):
     # Save weight_j_i and bias_j
    np.save("wji.npy", wji)
    np.save("bias_j.npy", bias_j)

    # Save weight_k_j and bias_k
    np.save("wkj.npy", wkj)
    np.save("bias_k.npy", bias_k)
