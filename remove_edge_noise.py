import numpy as np 
from PIL import Image

def remove_edge_noise(edge_img):
    width, height = edge_img.size 
    E = np.array(edge_img)

    M = np.zeros((height, width))
    N = np.zeros((height, width))  

    for i in range(0, height):
        for j in range(0, width): 
            if i-2 >= 0 and j-2 >= 0 and j+2 <= width-1:
                if E[i][j] == 255:
                    if(E[i-1][j-1] + E[i-1][j] + E[i-1][j+1] + E[i][j-1] > 0):
                        M[i][j] = max(M[i-1][j-1], M[i-1][j], M[i-1][j+1], M[i][j-1]) + 1
                    else:
                        M[i][j] = max(M[i-2][j-1], M[i-2][j], M[i-2][j+1], M[i][j-2], M[i-1][j+2], M[i][j-2]) + 1

    for i in reversed(range(height-1)):
        for j  in reversed(range(width-1)):
            if i+2 <= height-1 and j-2 >= 0 and j+2 <= width-1:
                if E[i][j] == 255:
                    if(E[i+1][j-1] + E[i+1][j] + E[i+1][j+1] + E[i][j+1] > 0):
                        N[i][j] = max(N[i+1][j-1], N[i+1][j], N[i+1][j+1], N[i][j+1]) + 1
                    else:
                        N[i][j] = max(N[i+2][j-1], N[i+2][j], N[i+2][j+1], N[i+1][j-2], N[i+1][j+2], N[i][j+2]) + 1

    # T_long = 100
    # T_short = 15
    
    # If car plate is further away, the lines will be shorter
    T_long = 70
    T_short = 8

    for i in range(0, height):
        for j  in range(0, width):
            if E[i][j] == 255:
                if (M[i][j]+N[i][j] > T_long or M[i][j] + N[i][j] < T_short):
                    E[i][j] = 0 
    
    return Image.fromarray(E)