## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, classify.py
## 3. In this challenge, you are only permitted to import numpy and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import math
#It's kk to import whatever you want from the local util module if you would like:
#from util.X import ... 

def classify(im):
    '''
    Example submission for coding challenge. 

    Args: im (nxmx3) unsigned 8-bit color image 
    Returns: One of three strings: 'brick', 'ball', or 'cylinder'

    '''


    #Let's guess randomly! Maybe we'll get lucky.
    labels = ['brick', 'ball', 'cylinder']
    #random_integer = np.random.randint(low = 0, high = 3)
    random_integer = 0;

    #find grayscale image
    im_scaled = im/255
    gray_im = convert_to_grayscale(im_scaled)

    #blur the image
    blur_image = blur_the_image(gray_im)

    #edge detection
    #sobel kernels
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    Gx = filter_2d(blur_image, Kx)
    Gy = filter_2d(blur_image, Ky)
    G = np.sqrt(Gx**2 + Gy**2)
    thresholded_img = (G > 0.035)

    #find out x and y coordinates
    y_coords, x_coords = np.where(thresholded_img)
    y_coords_flipped = thresholded_img.shape[1] - y_coords

    #calculate distances between multiple points with respect to center
    
    try:
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords_flipped)
        y_max = np.max(y_coords_flipped)
        dist_min_max = math.ceil((math.sqrt((x_max - x_min)**2 + (y_max - y_min)**2))/2)
        x1_index = np.min(np.where(x_coords == x_min));
        x1_y = y_coords_flipped[x1_index]
        x2_index = np.min(np.where(x_coords == x_max));
        x2_y = y_coords_flipped[x2_index]

        y1_index = np.min(np.where(y_coords_flipped == y_min));
        x_y1 = x_coords[y1_index]
        y2_index = np.min(np.where(y_coords_flipped == y_max));
        x_y2 = x_coords[y2_index]
        x_center = ((x_max - x_min)/2)+x_min
        y_center = ((y_max - y_min)/2)+y_min

        x_center_to_x_min = math.ceil(math.sqrt((x_min-x_center)**2 + (x1_y-y_center)**2))
        x_center_to_x_max = math.ceil(math.sqrt((x_max-x_center)**2 + (x2_y-y_center)**2))
        y_center_to_y_min = math.ceil(math.sqrt((x_y1-x_center)**2 + (y_min-y_center)**2))
        y_center_to_y_max = math.ceil(math.sqrt((x_y2-x_center)**2 + (y_max-y_center)**2))

        horizontal_dist = math.sqrt((x2_y-x1_y)**2 + (x_max-x_min)**2)
        vertical_dist = math.sqrt((y_max-y_min)**2 + (x_y2-x_y1)**2)

        x_coords_sorted = np.sort(x_coords);

        #first coordinate after x_min
        index_x1 = int(math.ceil((x_coords_sorted.shape[0])*1/8))
        x1_crd = x_coords_sorted[index_x1]
        index_x1_org = np.min(np.where(x_coords == x1_crd))
        y1_crd = y_coords_flipped[index_x1_org]
        index_x11_org = np.max(np.where(x_coords == x1_crd))
        y11_crd = y_coords_flipped[index_x11_org]

        #2nd coordinate after x_min
        index_x2 = int(math.ceil((x_coords_sorted.shape[0])*1/4))
        x2_crd = x_coords_sorted[index_x2]
        index_x2_org = np.min(np.where(x_coords == x2_crd))
        y2_crd = y_coords_flipped[index_x2_org]
        index_x21_org = np.max(np.where(x_coords == x2_crd))
        y21_crd = y_coords_flipped[index_x21_org]

        #3rd coordinate after x_min
        index_x3 = int(math.ceil((x_coords_sorted.shape[0])*3/8))
        x3_crd = x_coords_sorted[index_x3]
        index_x3_org = np.min(np.where(x_coords == x3_crd))
        y3_crd = y_coords_flipped[index_x3_org]
        index_x31_org = np.max(np.where(x_coords == x3_crd))
        y31_crd = y_coords_flipped[index_x31_org]

        #4th coordinate after x_min
        index_x4 = int(math.ceil((x_coords_sorted.shape[0])*1/2))
        x4_crd = x_coords_sorted[index_x4]
        index_x4_org = np.min(np.where(x_coords == x4_crd))
        y4_crd = y_coords_flipped[index_x4_org]
        index_x41_org = np.max(np.where(x_coords == x4_crd))
        y41_crd = y_coords_flipped[index_x41_org]

        #5th coordinate after x_min
        index_x5 = int(math.ceil((x_coords_sorted.shape[0])*5/8))
        x5_crd = x_coords_sorted[index_x5]
        index_x5_org = np.min(np.where(x_coords == x5_crd))
        y5_crd = y_coords_flipped[index_x5_org]
        index_x51_org = np.max(np.where(x_coords == x5_crd))
        y51_crd = y_coords_flipped[index_x51_org]

        #6th coordinate after x_min
        index_x6 = int(math.ceil((x_coords_sorted.shape[0])*3/4))
        x6_crd = x_coords_sorted[index_x6]
        index_x6_org = np.min(np.where(x_coords == x6_crd))
        y6_crd = y_coords_flipped[index_x6_org]
        index_x61_org = np.max(np.where(x_coords == x6_crd))
        y61_crd = y_coords_flipped[index_x61_org]

        x_center_to_x1_crd = math.ceil(math.sqrt((x1_crd-x_center)**2 + (y1_crd-y_center)**2))
        x_center_to_x11_crd = math.ceil(math.sqrt((x1_crd-x_center)**2 + (y11_crd-y_center)**2))
        x_center_to_x2_crd = math.ceil(math.sqrt((x2_crd-x_center)**2 + (y2_crd-y_center)**2))
        x_center_to_x21_crd = math.ceil(math.sqrt((x2_crd-x_center)**2 + (y21_crd-y_center)**2))
        x_center_to_x3_crd = math.ceil(math.sqrt((x3_crd-x_center)**2 + (y3_crd-y_center)**2))
        x_center_to_x31_crd = math.ceil(math.sqrt((x3_crd-x_center)**2 + (y31_crd-y_center)**2))
        x_center_to_x4_crd = math.ceil(math.sqrt((x4_crd-x_center)**2 + (y4_crd-y_center)**2))
        x_center_to_x41_crd = math.ceil(math.sqrt((x4_crd-x_center)**2 + (y41_crd-y_center)**2))
        x_center_to_x5_crd = math.ceil(math.sqrt((x5_crd-x_center)**2 + (y5_crd-y_center)**2))
        x_center_to_x51_crd = math.ceil(math.sqrt((x5_crd-x_center)**2 + (y51_crd-y_center)**2))
        x_center_to_x6_crd = math.ceil(math.sqrt((x6_crd-x_center)**2 + (y6_crd-y_center)**2))
        x_center_to_x61_crd = math.ceil(math.sqrt((x6_crd-x_center)**2 + (y61_crd-y_center)**2))

        check_arr = np.array([x_center_to_x_min, x_center_to_x_max, y_center_to_y_min, y_center_to_y_max,x_center_to_x1_crd,x_center_to_x11_crd,x_center_to_x2_crd,x_center_to_x21_crd,x_center_to_x3_crd,x_center_to_x31_crd,x_center_to_x4_crd,x_center_to_x41_crd,x_center_to_x5_crd,x_center_to_x51_crd,x_center_to_x6_crd,x_center_to_x61_crd])
        check_arr_avg = int(np.average(check_arr))
        check_arr_avg1 = check_arr_avg+5
        check_arr_avg2 = check_arr_avg-5

        compare_arr = np.logical_and(check_arr>=check_arr_avg2,check_arr<=check_arr_avg1)
        found_items = (compare_arr==1).sum()

        found_percent = (found_items/len(check_arr))*100
        check_arr_std = np.std(check_arr)

        if(check_arr_std>10 and (found_percent >= 0 or found_percent <= 40 )):
            random_integer = 0
        elif(check_arr_std>=1.5 and check_arr_std <= 4 and found_percent>=90):
            random_integer = 2
        else:
            random_integer = 1

        
    except:
        pass    
    
    return labels[random_integer]

def convert_to_grayscale(im):
    '''
    Convert color image to grayscale.
    Args: im = (nxmx3) floating point color image scaled between 0 and 1
    Returns: (nxm) floating point grayscale image scaled between 0 and 1
    '''
    gray = np.mean(im, axis = 2)
    return gray

def blur_the_image(im):
    kernel = np.array([[0.111, 0.111, 0.111],
               [0.111, 0.111, 0.111],
               [0.111, 0.111, 0.111]])
    M = kernel.shape[0] 
    N = kernel.shape[1]
    H = im.shape[0]
    W = im.shape[1]

    blur_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    for i in range(blur_image.shape[0]):
        for j in range(blur_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            blur_image[i, j] = (np.sum(np.multiply(image_patch, kernel)))/9
            
    return blur_image

#find edges of image - sobel operator
def filter_2d(im, kernel):
    '''
    Filter an image by taking the dot product of each 
    image neighborhood with the kernel matrix.
    Args:
    im = (H x W) grayscale floating point image
    kernel = (M x N) matrix, smaller than im
    Returns: 
    (H-M+1 x W-N+1) filtered image.
    '''
    M = kernel.shape[0] 
    N = kernel.shape[1]
    H = im.shape[0]
    W = im.shape[1]
    
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            
    return filtered_image