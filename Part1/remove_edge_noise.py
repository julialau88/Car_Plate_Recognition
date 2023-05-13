import numpy as np 
from PIL import Image

"""
Remove edge noises from edge image 
@param edge_img: edge image 
@param window_size: the window size (height, width) to slide over image when locating car plate, which is also an estimate of the car plate size 
@return the edge image with edge noises removed
"""
def remove_edge_noise(edge_img, window_size):
    width, height = edge_img.size 
    E = np.array(edge_img)

    M = np.zeros((height, width))
    N = np.zeros((height, width))  

    # Scan image from left to right and top to bottom 
    for i in range(0, height):
        for j in range(0, width): 
            # Calculate edge lengths away from the top starting from the left 
            if i-2 >= 0 and j-2 >= 0 and j+2 <= width-1:
                if E[i][j] == 255:
                    if(E[i-1][j-1] + E[i-1][j] + E[i-1][j+1] + E[i][j-1] > 0):
                        M[i][j] = max(M[i-1][j-1], M[i-1][j], M[i-1][j+1], M[i][j-1]) + 1
                    else:
                        M[i][j] = max(M[i-2][j-1], M[i-2][j], M[i-2][j+1], M[i][j-2], M[i-1][j+2], M[i][j-2]) + 1

    # Scan image from right to left and bottom to top 
    for i in reversed(range(height-1)):
        for j  in reversed(range(width-1)):
            # Calculate edge lengths away from the bottom starting from the right 
            if i+2 <= height-1 and j-2 >= 0 and j+2 <= width-1:
                if E[i][j] == 255:
                    if(E[i+1][j-1] + E[i+1][j] + E[i+1][j+1] + E[i][j+1] > 0):
                        N[i][j] = max(N[i+1][j-1], N[i+1][j], N[i+1][j+1], N[i][j+1]) + 1
                    else:
                        N[i][j] = max(N[i+2][j-1], N[i+2][j], N[i+2][j+1], N[i+1][j-2], N[i+1][j+2], N[i][j+2]) + 1

    
    # If car plate is further away, the lines will be shorter, vice versa
    # So, we use the estimated car plate size (window_size) to find the upper and lower bounds of the edge length
    T_long = window_size[0]
    T_short = window_size[0] * 0.1

    # Add up the two lengths obtained from the scans above 
    for i in range(0, height):
        for j  in range(0, width):
            if E[i][j] == 255:
                # Omit edge if the length is too long or too short 
                if (M[i][j]+N[i][j] > T_long or M[i][j] + N[i][j] < T_short):
                    E[i][j] = 0 
    
    return Image.fromarray(E)