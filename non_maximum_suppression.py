import numpy as np

def non_maximum_suppression(Angle, M, height, width):
    S = np.zeros((height,width))
    E = np.zeros((height,width))
    result = np.zeros((height,width))

    ## Compute sector 
    for h in range(0, height):
        for w in range(0, width):
            if Angle[h][w] >= 337.5 and Angle[h][w] <= 22.5: 
                S[h][w] = 0
            elif Angle[h][w] >= 157.5 and Angle[h][w] <= 202.5: 
                S[h][w] = 0
            elif Angle[h][w] >= 22.5 and Angle[h][w] <= 67.5: 
                S[h][w] = 1
            elif Angle[h][w] >= 202.5 and Angle[h][w] <= 247.5: 
                S[h][w] = 1
            elif Angle[h][w] >= 67.5 and Angle[h][w] <= 112.5: 
                S[h][w] = 2
            elif Angle[h][w] >= 247.5 and Angle[h][w] <= 292.5: 
                S[h][w] = 2
            elif Angle[h][w] >= 112.5 and Angle[h][w] <= 157.5: 
                S[h][w] = 3
            elif Angle[h][w] >= 292.5 and Angle[h][w] <= 337.5: 
                S[h][w] = 3

    ## Non maximum suppression 
    for h in range(0, height):
        for w in range(0, width):
            if S[h][w] == 0:    
                # Examine horizontal pair 

                ## Checking edges first
                if w == 0:
                    # Just check right pixel
                    if M[h][w+1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else: 
                        E[h][w] = 0
                elif w == width - 1:
                    # Just check left pixel 
                    if M[h][w-1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else: 
                        E[h][w] = 0
                else:
                    # Check both sides 
                    if M[h][w-1] < M[h][w] and M[h][w+1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0

            if S[h][w] == 2:
                # Check vertical pair 
                ## Checking edges first
                if h == 0:
                    # Just check bottom pixel 
                    if M[h+1][w] < M[h][w]:
                        E[h][w] = M[h][w]
                    else: 
                        E[h][w] = 0 
                elif h == height-1:
                    # Check top pixel 
                    if M[h-1][w] < M[h][w]:
                        E[h][w] = M[h][w]
                    else: 
                        E[h][w] = 0 
                else:
                    # Check both sides 
                    if M[h+1][w] < M[h][w] and M[h-1][w] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0
            
            if S[h][w] == 1:
                # Check 45 degree pair 
                ## Checking edges first
                if w == 0: 
                    if h == 0:
                        E[h][w] = M[h][w]
                    elif h > 0:
                        # Check top right 
                        if M[h-1][w+1] < M[h][w]:
                            E[h][w] = M[h][w]
                        else:
                            E[h][w] = 0
                elif w == width - 1: 
                    if h == height-1:
                        E[h][w] = M[h][w]
                    else:
                        # Check bottom left 
                        if M[h+1][w-1] < M[h][w]:
                            E[h][w] = M[h][w]
                        else:
                            E[h][w] = 0
                elif h == 0 and w != 0 and w != width - 1:
                    # Check bottom left 
                    if M[h+1][w-1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0

                elif h == height - 1 and width != 0 and width != width -1:
                    # Check top right
                    if M[h-1][w+1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0
                
                else: 
                    # Check both  top right and bottom left 
                    if M[h-1][w+1] < M[h][w] and M[h+1][w-1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0
            if S[h][w] == 3:
                # Check 135 degree pair 

                ## Checking edges first
                if w == 0: 
                    if h == height - 1:
                        E[h][w] = M[h][w]
                    elif h < height - 1:
                        # Check bottom  right 
                        if M[h+1][w+1] < M[h][w]:
                            E[h][w] = M[h][w]
                        else:
                            E[h][w] = 0
                elif w == width - 1: 
                    if h == 0:
                        E[h][w] = M[h][w]
                    else:
                        # Check top left 
                        if M[h-1][w-1] < M[h][w]:
                            E[h][w] = M[h][w]
                        else:
                            E[h][w] = 0
                elif h == 0 and w != 0 and w != width - 1:
                    # Check bottom right 
                    if M[h+1][w+1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0

                elif h == height - 1 and width != 0 and width != width -1:
                    # Check top left
                    if M[h-1][w-1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0
                else: 
                    # Check both 
                    if M[h+1][w+1] < M[h][w] and M[h-1][w-1] < M[h][w]:
                        E[h][w] = M[h][w]
                    else:
                        E[h][w] = 0
