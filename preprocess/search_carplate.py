import numpy as np
from scipy.signal import convolve2d
import cv2
import math

"""
Main function of locate car plate. Will call flag_candidate and sort_candidate functions
@param window_size: estimated car plate size, which will be used to slide over image to locate car plate 
@param edge_img: the edge image 
@param greyscale_img: the original greyscale image 
@return None if car plate not detected, else return car plate starting postion 
"""
def search_carplate(window_size, edge_img, greyscale_img): 
    img_arr = np.array(edge_img)
    width, height = edge_img.size
    result = None 
    candidate_starting_point = []
    
    ## Minimum pixels is that there must be at least 13% of edge pixels in window 
    pixel_range = (window_size[0]*window_size[1]*0.13)
    # Find possible candidates
    candidate_starting_point = flag_candidate(pixel_range, window_size, img_arr, width, height)
    
    # If more than one candidate, sort, else return
    if len(candidate_starting_point) > 0:
            result = sort_candidate(candidate_starting_point, window_size, edge_img, greyscale_img)
    elif len(candidate_starting_point) ==  0:
            result = None
    return result

"""
Flag possible car plate candidates 
@param edge_img: pixel range: minimum number of pixels in window to be flagged as car plate
@param window_size: the window size (height, width) to slide over image when locating car plate, which is also an estimate of the car plate size 
@param img_arr: the edge image array
@param width: width of image
@param height: height of image 
@return an array of possible car plate candidates 
"""
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
            G = convolve2d(window, kernel, "same")
            
            # Count number of edges 
            number_of_edges = np.count_nonzero(G)

            # If more than min number of pixels, consider it as candidate
            if number_of_edges >= pixel_range:
                number_of_candidate += 1
                start_point = (h, i, number_of_edges)
                candidate_starting_point.append(start_point)

    return candidate_starting_point


"""
Filter out candidates 
@param candidate_starting_point: array of candidate starting points
@param window_size: estimated size of car plate
@param img: original image
@param greyscale_img the greyscale img
@return None if no carplate found, else return car plate starting coords
"""
def sort_candidate(candidate_starting_point, window_size, img, greyscale_img):
    img_width, img_height = img.size

    i = 0
    j = 1
    sum_width = 0 
    count = 0 

    filter_candidates_height = []

    # Average width to filter overlapping blocks
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
        
    # Array for filtered candidates 
    filter_candidates = []
    i = 0 
    
    # Average height to filter overlapping blocks
    while i < len(filter_candidates_height):
        j = i + 1
        sum_height = filter_candidates_height[i][0]
        count = 1
        same_width = False

        # If the width is the same, add to sum, and increment count 
        while j < len(filter_candidates_height) and filter_candidates_height[j][1] == filter_candidates_height[i][1]:
            sum_height += filter_candidates_height[j][0]
            count += 1
            j += 1
            same_width = True

        # Get average height
        ave_height = sum_height/count
        filter_candidates.append((math.floor(ave_height), filter_candidates_height[i][1])) 

        if same_width == True: 
            i = j
        else:
            i += 1
    
    # Sort by position, usually not skewed to edges of image 
    if len(filter_candidates) > 0:
        # Set the width and height range car plate can be in 
        width_range = (img_width*0.15, img_width*0.85)
        height_range = (img_height*0.15, img_height*0.85)
    
        i = 0
        filter_candidates_pos = []
        
        # Loop through candidates 
        while i < len(filter_candidates):
            # If position not within range, do not add to array 
            if filter_candidates[i][0] < height_range[0] or filter_candidates[i][0] > height_range[1] or filter_candidates[i][1] < width_range[0] or filter_candidates[i][1] > width_range[1]:
                i+=1
                continue
            else:
                filter_candidates_pos.append(filter_candidates[i])

            i+=1
    
    # Return if there is only one candidate left 
    else:
        return filter_candidates[0]
    
    if len(filter_candidates_pos) == 1:
        return filter_candidates_pos[0]
    
    ############################# Vertical Projection 
    # Otsu's
    greyscale_img_arr = np.array(greyscale_img)
    _, threshold_image = cv2.threshold(greyscale_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    gap_arr = []
    if len(filter_candidates_pos) > 1: 
        for i in range(0, len(filter_candidates_pos)):
            window_arr = threshold_image[filter_candidates_pos[i][0]: filter_candidates_pos[i][0] + window_size[0]]
            window_arr = window_arr[:, filter_candidates_pos[i][1]: filter_candidates_pos[i][1] + window_size[1]]

            # Vertical projection 
            vertical_projection = np.sum(window_arr, axis=0)
            vertical_projection = vertical_projection/255

            # Set thresholds for gap, character, and character width 
            gap_length = round(window_size[1]*0.10)     # How long the gap can be
            gap_thres = round(window_size[1]*0.06)      # Below this threshold, it is considered that there is a gao
            non_gap_thres = round(window_size[1]*0.09)  # Above this threshold, it is considered as a possible character 
            char_width_min = round(window_size[1]*0.03) # Minimum width of a character 
            char_width_max = round(window_size[1]*0.17) # 
        
            # Varibales 
            num_gap = 0
            num_char = 0
            gap_check = False
            col = 0 

            # Loop through verical projection array 
            while col < len(vertical_projection):

                if vertical_projection[col] >= non_gap_thres: 

                    # Find end of current char 
                    for j in range(col, len(vertical_projection)):
                        if vertical_projection[j] <= gap_thres:
                            # If the non-gap is within the possible width of a character, a possible character is found
                            if j - col >= char_width_min and j-col <= char_width_max:
                                num_char += 1
                                gap_check = True
                                col = j + 1
                                break
                            # If the non-gap is longer than a width of a possible char, not a possible character 
                            elif j-col > char_width_max or j-col < char_width_min:
                                col = j + 1
                                gap_check = False
                                break 
                            # End of vertical projection array
                            elif j == len(vertical_projection) - 1:
                                col = len(vertical_projection)
                                gap_check = False
                                break
                        # End of vertical projection array
                        elif j == len(vertical_projection) - 1:
                            col = len(vertical_projection)
                            gap_check = False
                            break

                    
                    # If a gap is found, find start of next possible char
                    if gap_check: 
                        for k in range(col+1, len(vertical_projection)):

                            if vertical_projection[k] >= non_gap_thres: 
                                # See gap width 
                                gap_width = k - col 
                                if gap_width <= gap_length:
                                    num_gap += 1
                                col = k 
                                break
                else:
                    col += 1

            tup = (num_gap, num_char)
            gap_arr.append(tup)

    # Variables 
    max_gap = 0
    car_plate_found = False

    for i in range(0, len(gap_arr)):
        # Cannot have more than 7 gaps in car plate or less than 2
        # Cannot have more than 7 characters in car plate or less than 2
        if gap_arr[i][0] >= 2 and gap_arr[i][0] <= 7 and gap_arr[i][1] >= 2 and gap_arr[i][1] <= 7: 
            car_plate_found = True
            # Find max numbers of gaps found 
            if gap_arr[i][0] > max_gap:
                max_gap = gap_arr[i][0]
    
    # Return the maximum
    for i in range(0, len(gap_arr)):
        if gap_arr[i][0] == max_gap and car_plate_found:
            return filter_candidates_pos[i]
    return None





