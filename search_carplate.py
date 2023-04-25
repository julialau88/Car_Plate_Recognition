from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
from subblocks import blockshaped
import math

def search_carplate(edge_img, greyscale_img): 
    img_arr = np.array(edge_img)
    width, height = edge_img.size
    window_size_arr = [(130, 335), (100, 275), (70, 220)]
    iter = 0 
    result = None 

    while iter < (len(window_size_arr)) and result == None:
        candidate_starting_point = []
        while iter < (len(window_size_arr)) and len(candidate_starting_point) < 1:
            # print("Check", window_size_arr[iter])
            pixel_range = (window_size_arr[iter][0]*window_size_arr[iter][1]*0.1)
            candidate_starting_point = flag_candidate(pixel_range, window_size_arr[iter], img_arr, width, height)
            print("candidate", candidate_starting_point)
            iter += 1
        
        if len(candidate_starting_point) > 0:
            result = sort_candidate(candidate_starting_point, window_size_arr[iter-1], edge_img, img_arr, greyscale_img)
        elif len(candidate_starting_point) ==  0:
            result = None

    return window_size_arr[iter-1], result

def flag_candidate(pixel_range, window_size, img_arr, width, height):
    number_of_candidate = 0
    candidate_starting_point = []
    kernel = np.ones((4,10))
    step_size_x = 15
    step_size_y = 15

    ## Loop through block by block and convolve
    for h in range(0, height - window_size[0], step_size_y):
        # Take out each row 30 pixels
        rows = img_arr[h:h+window_size[0]]
        for i in range(0, width-window_size[1], step_size_x):
            window = rows[:, i:i+window_size[1]]
            # Convolve 
            G = convolve2d(window, kernel, "same")
            number_of_edges = np.count_nonzero(G)

            if number_of_edges > pixel_range:
                number_of_candidate += 1
                start_point = (h, i, number_of_edges)

                candidate_starting_point.append(start_point)
             
    # print(number_of_candidate)
    # print(candidate_starting_point)

    return candidate_starting_point

def sort_candidate(candidate_starting_point, window_size, img, img_arr, greyscale_img):
    img_width, img_height = img.size

    i = 0
    j = 1
    sum_width = 0 
    count = 0 

    filter_candidates_height = []

    # (height , width)
    while i < len(candidate_starting_point):
        j = i + 1
        sum_width = candidate_starting_point[i][1]
        count = 1
        same_height = False
        while j < len(candidate_starting_point) and candidate_starting_point[j][0] == candidate_starting_point[i][0]:
            sum_width += candidate_starting_point[j][1]
            count += 1
            j += 1
            same_height = True

        ave_width = sum_width/count
        filter_candidates_height.append((candidate_starting_point[i][0], math.floor(ave_width))) 

        if same_height == True: 
            i = j
        else:
            i += 1
        
    # print("Ave_width",filter_candidates_height)
    
    filter_candidates = []
    i = 0 
    
    # Average height 
    while i < len(filter_candidates_height):
        j = i + 1
        sum_height = filter_candidates_height[i][0]
        count = 1
        same_width = False
        while j < len(filter_candidates_height) and filter_candidates_height[j][1] == filter_candidates_height[i][1]:
            sum_height += filter_candidates_height[j][0]
            count += 1
            j += 1
            same_width = True

        ave_height = sum_height/count
        filter_candidates.append((math.floor(ave_height), filter_candidates_height[i][1])) 

        if same_width == True: 
            i = j
        else:
            i += 1

    print("After filtering based on overlapping blocks",filter_candidates)
    
    # Sort by position, usually not skewed to the left or right
    if len(filter_candidates) > 0:
        width_range = (img_width*0.15, img_width*0.85)
        height_range = (img_height*0.15, img_height*0.85)
    
        i = 0
        filter_candidates_pos = []
        while i < len(filter_candidates):
            if filter_candidates[i][0] < height_range[0] or filter_candidates[i][0] > height_range[1] or filter_candidates[i][1] < width_range[0] or filter_candidates[i][1] > width_range[1]:
                i+=1
                continue
            else:
                filter_candidates_pos.append(filter_candidates[i])

            i+=1
            
        print("After filtering with positions", filter_candidates_pos)
    else:
        return filter_candidates[0]
    
    if len(filter_candidates_pos) == 1:
        return filter_candidates_pos[0]
    
    ############################# Vertical Projection 
    # cv2_gray_scale = cv2.cvtColor(greyscale_img, cv2.COLOR_BGR2GRAY)
    greyscale_img_arr = np.array(greyscale_img)
    _, threshold_image = cv2.threshold(greyscale_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("Threshold_image", threshold_image)
    # cv2.waitKey(0)
    gap_arr = []
    if len(filter_candidates_pos) > 1: 
        for i in range(0, len(filter_candidates_pos)):
            window_arr = threshold_image[filter_candidates_pos[i][0]: filter_candidates_pos[i][0] + window_size[0]]
            window_arr = window_arr[:, filter_candidates_pos[i][1]: filter_candidates_pos[i][1] + window_size[1]]
            # cv2.imshow("Threshold_image", window_arr)
            # cv2.waitKey(0)
            vertical_projection = np.sum(window_arr, axis=0)
            vertical_projection = vertical_projection/255

            print(vertical_projection)

            ## Show vertical projection histogram 
            # blankImage = np.zeros_like(window_arr)
            # for i, value in enumerate(vertical_projection):
            #     cv2.line(blankImage, (i, 0), (i, window_size[0]-int(value)), (255, 255, 255), 1)
            # cv2.imshow("New Histogram Projection", blankImage)
            # cv2.waitKey(0)

            # Find the gaps between charcters 
            # Set a threshold to the gap 
            # Must convert in terms of ratio!!
            gap_length = round(window_size[1]*0.15)
            
            gap_thres = round(window_size[1]*0.06)
            non_gap_thres = round(window_size[1]*0.08)

            char_length_min = round(window_size[1]*0.03)
            char_length_max = round(window_size[1]*0.20)
            
            # gap_length = round(window_size[1]*0.15)
            
            # gap_thres = round(window_size[1]*0.06)
            # non_gap_thres = round(window_size[1]*0.09)

            # char_length_min = round(window_size[1]*0.04)
            # char_length_max = round(window_size[1]*0.10)

            print(gap_length, gap_thres, non_gap_thres, char_length_min, char_length_max)
            num_gap = 0
            num_char = 0
            gap_check = False

            # Find possible gaps within 
            col = 0 
            while col < len(vertical_projection):
                # print(col)
                if vertical_projection[col] >= non_gap_thres: 

                    # Start 
                    # Find end of current char 
                    for j in range(col, len(vertical_projection)):
                        if vertical_projection[j] <= gap_thres:
                            # print("gap_break", j, col)
                            
                            if j - col >= char_length_min and j-col <= char_length_max:
                                # Found a possible char
                                num_char += 1
                                gap_reached = j 
                                gap_check = True
                                col = j + 1
                                break 
                            elif j-col > char_length_max or j-col < char_length_min:
                                # Longer than a width of a possible char 
                                col = j + 1
                                gap_check = False
                                break 
                            elif j == len(vertical_projection) - 1:
                                col = len(vertical_projection)
                                gap_check = False
                                break
                        elif j == len(vertical_projection) - 1:
                            col = len(vertical_projection)
                            gap_check = False
                            break

                    
                    # Find start of next possible char
                    if gap_check: 
                        for k in range(col+1, len(vertical_projection)):
                            if vertical_projection[k] >= non_gap_thres: 
                                # See gap width 
                                gap_width = k - col 
                                print("gap_width", gap_width)
                                if gap_width <= gap_length:
                                    num_gap += 1
                                col = k 
                                break
                else:
                    col += 1

            # print(num_gap, num_char)
            tup = (num_gap, num_char)
            gap_arr.append(tup)

    print(gap_arr)
    total_arr = []
    max_gap = 0
    max_char  = 0 
    for i in range(0, len(gap_arr)):
        if gap_arr[i][0] >= 2 and gap_arr[i][0] <= 6 and gap_arr[i][1] >= 2 and gap_arr[i][1] <= 7: 
            if gap_arr[i][0] > max_gap:
                max_gap = gap_arr[i][0]
            if gap_arr[i][1] > max_char:
                max_char = gap_arr[i][1]

    print("max gap", max_gap)
    print("max_char", max_char)

    for i in range(0, len(gap_arr)):
        if gap_arr[i][0] == max_gap and gap_arr[i][1] == max_char:
            print("result", filter_candidates_pos[i])
            return filter_candidates_pos[i]
    return None





