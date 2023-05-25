import numpy as np 
# wji = [[0.1, 0.2], [0.3, 0.4]]
# np.save("wji.npy", wji)

try: 
    wji = np.load("wji.npy")
    wkj = np.load("wkj.npy")
    bias_j = np.load("bias_j.npy")
    bias_k = np.load("bias_k.npy")
except FileNotFoundError:
    print("INITIALISE ONCE")
# print(wji)