import numpy as np 
from scipy.signal import convolve2d

def sobel(width, height, img_arr):
    Kx = np.matrix("-1, -2, -1; 0, 0, 0;, 1, 2, 1")
    Ky = np.matrix("-1, 0, 1; -2, 0, 2;, -1, 0, 1")
    # Apply the Sobel operator
    Gx = convolve2d(img_arr, Kx, "same")
    Gy = convolve2d(img_arr, Ky, "same")
    G = np.sqrt(Gx**2 + Gy**2)

    # Clip range 0-255
    G = np.clip(G, 0, 255)
    T = np.percentile(G, 85)
    print(G)
    print(T)

    for x in range(0, width):
        for y in range(0, height):
            if G[x][y] >= T: 
                G[x][y] = 1
            else: 
                G[x][y] = 0

    return G
            