from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

def search_carplate(pixel_range, window_size, edge_img, greyscale_img): 
    img_arr = np.array(edge_img)
    img_arr[img_arr==255] = 1
    width, height = edge_img.size
    candidate_starting_point = flag_candidate(pixel_range, window_size, img_arr, width, height)
    sort_candidate(candidate_starting_point, window_size, edge_img, img_arr, greyscale_img)

def flag_candidate(pixel_range, window_size, img_arr, width, height):
    number_of_candidate = 0
    candidate_starting_point = []
    # kernel = np.ones((window_size))
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
            # print(number_of_pixels)
            # print(G)
        # break
             
    print(number_of_candidate)
    print(candidate_starting_point)

    return candidate_starting_point

def sort_candidate(candidate_starting_point, window_size, img, img_arr, greyscale_img):
    img_width, img_height = img.size
    h = window_size[0]
    w = window_size[1]
    # How to know if it is a carplate 
    i = 0
    j = 0 

    filter_candidates = []
    while i < len(candidate_starting_point):
        # Take first point 
        filter_candidates.append(candidate_starting_point[i])

        while j < len(candidate_starting_point):
            if candidate_starting_point[j][0] - candidate_starting_point[i][0] <= h:
                j = j+1
            else:
                break 
        i += j

    # Sort by position, usually not skewed to the left or right
    width_range = (img_width*0.15, img_width*0.85)
    height_range = (img_height*0.15, img_height*0.85)
    
    i = 0
    filter_candidates_pos = []
    while i < len(filter_candidates):
        if filter_candidates[i][0] < height_range[0] or filter_candidates[i][0] > height_range[1] or filter_candidates[i][1] < width_range[0] or filter_candidates[i][1] > width_range[1]:
            continue
        else:
            filter_candidates_pos.append(filter_candidates[i])

        i+=1
        
    print(filter_candidates_pos)

    # Carplate has a sequence of evenly spaced characters 
    # Vertical Projection 
    # test_arr = img_arr[filter_candidates_pos[1][0]: filter_candidates_pos[1][0] + window_size[0]]
    # test_arr = img_arr[:, filter_candidates_pos[1][1]: filter_candidates_pos[1][1] + window_size[1]]
    # vertical_projection = np.sum(test_arr, axis=0)
    # print(img_arr)
    # print(vertical_projection)

    # blankImage = np.zeros_like(test_arr)
    # for i, value in enumerate(vertical_projection):
    #     cv2.line(blankImage, (i, 0), (i, img_height-int(value)), (255, 255, 255), 1)

    # cv2.imshow("New Histogram Projection", blankImage)

    # cv2.waitKey(0)

    # Tend to have a sequence of lines in vertical direction 

    ## Grey scale colour contrast
    # g_hist = img.histogram()
    img_arr = np.array(greyscale_img)
    print(img_arr)
    test_arr = img_arr[filter_candidates_pos[0][0]: filter_candidates_pos[0][0] + window_size[0]]
    test_arr = img_arr[:, filter_candidates_pos[0][1]: filter_candidates_pos[0][1] + window_size[1]]
    test_img = Image.fromarray(test_arr)
    g_hist = test_img.histogram()
    print(g_hist)   
    plt.figure(0)
    for i in range(len(g_hist)):
        plt.bar(i, g_hist[i])
    plt.show()

# def flag_candidate(pixel_range, window_size, img_arr, width, height):

#     number_of_candidate = 0
#     candidate_starting_point = []

#     ## Slide window through whole image 
#     ## Window size
#     for h in range(0, height - window_size[0]):
#         rows = img_arr[h:h+window_size[0]]
#         # Take out each 80 columns 
#         for i in range(0, width - window_size[1]):
#             window = rows[:, i:i+window_size[1]]
#             number_of_pixels = np.count_nonzero(window)

#             if number_of_pixels >= pixel_range:
#                 number_of_candidate += 1
#                 start_point = (h, i)
#                 candidate_starting_point.append(start_point)
#     print(number_of_candidate)
#     print(candidate_starting_point)


