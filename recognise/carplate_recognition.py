"""
Automated segmentation with car plate recognition
"""
from PIL import Image, ImageOps
import numpy as np
import cv2
from ann import Forward_Hidden_Output, Forward_Input_Hidden
import numpy as np
import cv2
from PIL import Image, ImageOps

print("RECOGNISING CAR PLATES")
for k in range(1, 11):
    print("CAR PLATE", k)
    
    # Read file and convert to grey scale
    file = "Part2/Dataset/CarPlate/" + str(k) +".jpg"
    img = Image.open(file).convert('L')
    img = ImageOps.exif_transpose(img) # Make sure image remains in original orientation

    # Image resize 
    size = (900,  900)  
    img =  img.resize(size)

    img_arr = np.array(img) # Image array 
    greyscale_img = img.resize(size) # Greyscale image
    target_values = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W", "0","1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Otsu method
    greyscale_img_arr = np.array(greyscale_img)
    _, threshold_image = cv2.threshold(greyscale_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # output either 0 or 255

    # Vertical projection
    vertical_projection = np.sum(threshold_image, axis=0)
    vertical_projection = vertical_projection/255

    # Set a threshold
    threshold = 0.12 * np.max(vertical_projection)

    # Detect gaps
    gaps = np.where(vertical_projection < threshold)[0]

    # Initialise variables
    segments = []
    start = 0

    # Segment characters
    for gap in gaps:  
        # Check if there is character
        if gap - start > 1: 
            char_segment = greyscale_img_arr[:, start-13:gap+13]  # extract the character
            segments.append(char_segment)  # save it to array
        start = gap + 1

    string = ""
    for i, segment in enumerate(segments):
        if segment.size > 0:
            input_img = Image.fromarray(segment)
            w,h = input_img.size

            # Image resize 
            size = (50,40)   
            input_img =  input_img.resize(size)

            input_img_arr = np.array(input_img) 
            width, height = input_img.size

            # Input_Neurons: Initialise total number of neurons
            Input_Neurons = width*height

            # Hidden_Neurons: Initialise number of hidden neurons
            Hidden_Neurons = 100

            # Output_Neurons: Initialise number of output neurons
            Output_Neurons = len(target_values)
       
            # Otsu's
            _, threshold_image = cv2.threshold(input_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            input_arr = np.divide(threshold_image, 255)
            input_arr = input_arr.flatten()

            # Load weights
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

            max_index = np.argmax(OutK)
            string += target_values[max_index]

    # Print car plate
    print("THE RECOGNISED CARPLATE IS:", string)
