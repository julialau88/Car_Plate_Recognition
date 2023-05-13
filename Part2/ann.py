import numpy as np 
from PIL import Image
from scipy.signal import convolve2d
import math
from sobel import sobel

# Step 1: initialisation of weights 
def Weight_Initialization(Input_Neurons, Hidden_Neurons, Output_Neurons):
    # Initializing of the Weights.
    # Random float number between -0.5 to 0.5.
    wji= np.random.uniform(-0.5, 0.5, size=(Hidden_Neurons, Input_Neurons))
    wkj = np.random.uniform(-0.5, 0.5, size=(Output_Neurons, Hidden_Neurons))
    bias_j = np.random.uniform(0, 1, size=(Hidden_Neurons, 1))          # In this format: [[0.32650096][0.16459484][0.07902434][0.28043552][0.68215983]]
    bias_k = np.random.uniform(0, 1, size=(Output_Neurons, 1))
    return wji, wkj, bias_j, bias_k

# Step 2. Reading of Training Files, and Target Files.
def read_files(file):
    # Reading of Input File, and Target File.
    # Ask the user to enter the number of input, hidden and output neurons.
    input_img = Image.open(file).convert('L')
    input_img_arr = np.array(input_img) 
    width, height = input_img.size

    return input_img_arr, width, height

# Step 3. Forward Propagation from Input -> Hidden Layer.
def Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_img_arr, bias_j, NetJ, OutJ, wji):

    # Obtain the results at each neuron in the hidden layer
    # Initialise NetJ --> same size as hidden neurons 

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

# Step 4. Forward Propagation from Hidden -> Output Layer.
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
    for k in range(0,Output_Neurons):
        OutK[k] = 1/(1 + math.exp(-(NetK[k])))
    
    return NetK, OutK

# Step 5. Check for the global error or number of iterations.
def Check_for_End(target_arr, iter, error_threshold):
    # Check whether the total error is less than the error set by the user or the number of iterations is reached.
    # returns true or false
    # If TRUE, proceed to Step 10
    return 0 

def main():
    ### Initialisation of parameters 
    file = "Enter_file_here"
    
    # Input_Neurons: Initialise total number of neurons
    Input_Neurons = width*height

    # Hidden_Neurons: Initialise number of hidden neurons
    Hidden_Neurons = 15

    # Output_Neurons: Initialise number of output neurons
    Output_Neurons = 3

    # Global error

    # Target here is the expected output 
    target_arr = [1,0,0] # Target arr now should be of length 26 (alphabets)+ 10 (numeric)
    
    input_img_arr, width, height = read_files(file)
    input_img_arr = sobel(width, height, input_img_arr)
    input_img_arr = input_img_arr.flatten()
    wji, wkj, bias_j, bias_k = Weight_Initialization(Input_Neurons, Hidden_Neurons, Output_Neurons)

    NetJ = np.zeros(Hidden_Neurons)
    OutJ = np.zeros(Hidden_Neurons)
    NetJ, OutJ = Forward_Input_Hidden(Input_Neurons, Hidden_Neurons, input_img_arr, bias_j, wji)

    NetK = np.zeros(Output_Neurons)
    OutK = np.zeros(Output_Neurons)
    NetK, OutK = Forward_Hidden_Output(wkj, Output_Neurons, Hidden_Neurons, OutJ, bias_k, NetK, OutK)

    # wji, wkj, bias_j, bias_k = Weight_Initialization(5, 5, 5)
    # print(bias_k)