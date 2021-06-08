# %% 
print("Lets Get This Party Started")
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# %% Camera Calibration - Generate Camera Matrix and Distortion Coefficients
def cameracalibration_gen():
    # Prepare Object Points: Points referenced to actual object in distorted image
    # Object Points: 3D points in real world space
    # Image Points: 2D points in image plane
    # Calibration setup
    nx = 9 #Calibration Images: Number of inside corners in x
    ny = 5 #Calibration Images: Number of inside corners in y

    # Define "Standard Object Points"
    points_obj_std = np.zeros((nx*ny,3),np.float32)
    points_obj_std[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) 
    img_size = (720,1280)
    #print('points_obj_std',points_obj_std)

    # Create empty arrays
    points_object = []
    points_image = []

    #Create list of calibration images
    images_calibration = glob.glob('../camera_cal/calibration*.jpg')
    #print('images_calibration',images_calibration)

    # Loop through calibration image list and extract chessboard corner data
    for idx, fname in enumerate(images_calibration):
        # Read in image
        img = cv2.imread(fname)
        # Convert to grayscale
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)
        # Record corners if found
        if ret == True:
            print("ret=",idx)
            # Append Object Points (standard array) and Image Points (extracted corners)
            points_object.append(points_obj_std)
            points_image.append(corners)
            # Display Corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            write_name = '../output_images/img_res_calibration_%s'+str(img)+'.jpg'
            cv2.imwrite(write_name, img)
    print('Camera Calibration Completed')

    # Generate camera calibration files
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points_object, points_image, img_size,None,None)
    
    # Save camera calibration data in "Pickle" file
    cal_pickle = {}
    cal_pickle["mtx"] = mtx
    cal_pickle["dist"] = dist
    pickle.dump( cal_pickle, open( "camera_calibration_pickle.p", "wb" ) )

    #return mtx, dist

# %% Camera Calibration - Undistort Image
def cameracalibration_undistort(img):
    
    #img_tst = cv2.imread('../camera_cal/test_calibration_2.jpg')
    #img_tst = cv2.imread('../test_images/test4.jpg')
    #img_size = (img_tst.shape[0],img_tst.shape[1])
    #img_size = (720,1280)
    #print(img_size)

    # Load camera calibration data from "Pickle" file
    cal_pickle = pickle.load(open( "camera_calibration_pickle.p", "rb" ) )
    mtx = cal_pickle["mtx"]
    dist = cal_pickle["dist"]

    # Undistort test image
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)

    #Visualize undistortion 
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    #ax1.imshow(img)
    #ax1.set_title('Original Image', fontsize=30)
    #ax2.imshow(img_undist)
    #ax2.set_title('Undistorted Image', fontsize=30)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    return img_undist

# %% Camera Calibration Initialize and Test
# Generate Camera Matrix and Distortion Coefficients
cameracalibration_gen()

# Load Test Image
img_tst = cv2.imread('../camera_cal/calibration1.jpg')

# Undistort Image
img_undist = cameracalibration_undistort(img_tst)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img_tst)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(img_undist)
ax2.set_title('Undistorted Image', fontsize=30)


# %% Color/gradient threshold
def colorgradthreshold(img, s_channel_thresh=(170, 255), sobelx_thresh=(20, 100)):
    #img = np.copy(img)
    
    # Convert to HLS color space and separate channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0] # Hue
    l_channel = hls[:,:,1] # Light
    s_channel = hls[:,:,2] # Saturation
    #plt.figure()
    #plt.imshow(h_channel)
    #plt.figure()
    #plt.imshow(l_channel)
    #plt.figure()
    #plt.imshow(s_channel)
    #cv2.imwrite('../test_images/tst_colthresh_h_chan.jpg',h_channel)
    #cv2.imwrite('../test_images/tst_colthresh_l_chan.jpg',l_channel)
    #cv2.imwrite('../test_images/tst_colthresh_s_chan.jpg',s_channel)

    # Convert original image to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img_gray)
    #cv2.imwrite('../test_images/tst_colthresh_grey.jpg',img_gray)

    ## Gradient Threshold
    # Run Sobel - Derivative in x direction - on grayscale (or s_channel?) 
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # Take the derivative of grayscale or ?? channel
    sobelx_abs = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    sobelx_scaled = np.uint8(255*sobelx_abs/np.max(sobelx_abs))
    # Threshold x gradient binary
    sobelx_binary = np.zeros_like(sobelx_scaled)
    sobelx_binary[(sobelx_scaled >= sobelx_thresh[0]) & (sobelx_scaled <= sobelx_thresh[1])] = 1
    
    ## Colour Threshold 
    # Threshold "saturation" channel binary
    s_channel_binary = np.zeros_like(s_channel)
    s_channel_binary[(s_channel >= s_channel_thresh[0]) & (s_channel <= s_channel_thresh[1])] = 1
    
    #plt.figure()
    #plt.imshow(s_channel_binary)
    #cv2.imwrite('../test_images/tst_colthresh_s_chan_bin.jpg',s_channel_binary)
    
    # Stack each channel  (Red / Green / Blue) to view individual contributions in green & blue
    color_binary = np.dstack(( np.zeros_like(sobelx_binary), sobelx_binary, s_channel_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(s_channel_binary == 1) | (sobelx_binary == 1)] = 1
    combined_binary =  combined_binary * 255
    # print(color_binary.shape,combined_binary.shape)

    return combined_binary
    #return color_binary, combined_binary
    #combined_binary_reshape = np.expand_dims(combined_binary,axis=2)
    #combined_binary_reshape = np.dstack((combined_binary_reshape,combined_binary_reshape,combined_binary_reshape))
    #print(combined_binary_reshape.shape)
    #return combined_binary_reshape
    
#img_tst_cgt = cv2.imread('../test_images/test5.jpg')
#print(img_tst_cgt.shape)
#result_com  = colorgradthreshold(img_tst_cgt)
#print(result_com.shape)
# #Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img_tst_cgt)
#ax1.set_title('Original Image', fontsize=40)
#ax2.imshow(result_com,cmap='gray')
#ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# # Write Output Image
#cv2.imwrite('../test_images/tst_colthresh.jpg',result_col)
#cv2.imwrite('../test_images/tst_comthresh.jpg',result_com)

# %% Perspective Transform

def perspectivetransform(img,points_source,points_dest):
    # Define Image and Object Points
    #points_object = np.float32([[200,720],[590,450],[700,450],[1075,720]])
    #points_image = np.float32([[300,720],[300,0],[975,0],[975,720]])
    points_object = points_source
    points_image = points_dest

    img_shape = img.shape[1],img.shape[0]
    
    # Calculate Transfrom Matrix
    M = cv2.getPerspectiveTransform(points_object,points_image)
    # Calculate Inverse Transform Matrix
    Minv = cv2.getPerspectiveTransform(points_image,points_object)
    # Generate warped image
    img_warped = cv2.warpPerspective(img,M,img_shape,cv2.INTER_LINEAR)

    # Return warped image
    return img_warped, M, Minv 

# %% Detect Lane Lines
def find_lane_pixels(img_binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_binary_warped[img_binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)

    #print("histogram:",histogram, "size",histogram.shape)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img_binary_warped, img_binary_warped, img_binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Calculate vehicle offset from lane center (in pixels)
    center_offset_pixels = ((rightx_base-leftx_base)-midpoint) #np.absolute 
    #print("mid:",midpoint,"leftx:",leftx_base,"rightx:",rightx_base,"offset:",center_offset_pixels)

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9 #8 #5
    # Set the width of the windows +/- margin
    margin = 75 #100 #125 #150
    # Set minimum number of pixels found to recenter window
    minpix = 50 #100 #50

    # Set height of windows - based on nwindows above and image shape (height)
    window_height = int(img_binary_warped.shape[0]//nwindows)
    #print("win_height:",window_height)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_binary_warped.nonzero() # 2d array of x /y coordinate position of nonzero pixels
    nonzeroy = np.array(nonzero[0]) # y coordinate position of nonzero pixel
    nonzerox = np.array(nonzero[1]) # x coordinate position of nonzero pixel
    #print("nonzero warped :",nonzero,"nonzero warped size :",len(nonzero),"nonzero x:",nonzerox,"size:",nonzerox.shape,"nonzero y:",nonzeroy,"size:",nonzeroy.shape)
    # nonzero is a 2D turple with a total of 89486 x 89486 indices indicating non-zero pixel coordinates
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        # Window high / low along y axis
        win_y_low = img_binary_warped.shape[0] - (window+1)*window_height
        win_y_high = img_binary_warped.shape[0] - window*window_height
        
        # Window high / low along x axis for left and right lanes
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current-margin
        win_xright_high = rightx_current+margin
        #print("binary warped shape:",img_binary_warped.shape,"winy low:",win_y_low,"winy high:",win_y_high)
        #print("binary warped shape:",win_xleft_low,win_xleft_high,win_xright_low,win_xright_high)
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2) 
        
        # Identify the nonzero pixels in x and y within the window      
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) 
        & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) 
        & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        #print ("good_left:",good_left_inds,"size:",good_left_inds.shape,"good_right:",good_right_inds,"size:",good_right_inds.shape)
        # Why are we extracting only the [0] index and why are the values greater than 1280?
        # Ans1: Nonzero() function returns a truple; Adding the [0] seems to convert the turple into and array of size 1
        # Ans2: Recall we are looking at indicies of non-zero pixels. Therefor the max value is the total number of pixels (720 * 1280 = 921600)
        # That said, the total number should not exceed the number of non zer pixels detected and stored in nonzero (89486 x 89486) 
        #numpy.nonzero()function is used to Compute the indices of the elements that are non-zero.
        #It returns a tuple of arrays, one for each dimension of arr, containing the indices of the non-zero elements in that dimension.
        #The corresponding non-zero values in the array can be obtained with arr[nonzero(arr)] . To group the indices by element, 
        #rather than dimension we can use transpose(nonzero(arr)).

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter window if # pixels > minpix pixels; If less, lane line lost, keep same x pos
        ### (`right` or `leftx_current`) on their mean position ###
        if (len(good_left_inds) > minpix):
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            #print("lefx_current:",leftx_current)
        if (len(good_right_inds) > minpix):
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            #print("rightx_current:",rightx_current)
            #notice we are only interested in the x values
            #Get actual x non zero position values by feeding good_left_inds into nonzerox array
    
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    # How do indices relate to the x or y pixel positions?
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # print("leftx:",leftx,"left_lane_inds:",left_lane_inds)
    return out_img, leftx, lefty, rightx, righty, center_offset_pixels

def search_lanes_sliding_win(img_binary_warped):
    # Find our lane pixels first
    out_img, leftx, lefty, rightx, righty, center_offset_pixels = find_lane_pixels(img_binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_polyfit = np.polyfit(lefty,leftx,2)
    right_polyfit = np.polyfit(righty,rightx,2)
    #print("left_fit:",left_fit,"leftx:",leftx,"lefty:",lefty)
    #Recall: while normally you calculate a y-value for a given x, here we 
    # do the opposite. Why? Because we expect our lane lines to be (mostly) 
    # vertically-oriented

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_binary_warped.shape[0]-1, img_binary_warped.shape[0] )
    try:
        left_fitx = left_polyfit[0]*ploty**2 + left_polyfit[1]*ploty + left_polyfit[2]
        right_fitx = right_polyfit[0]*ploty**2 + right_polyfit[1]*ploty + right_polyfit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines -Uncomment to see polynomial fit overlay on image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #print(ploty.shape,ploty)
    #cv2.imwrite('../test_images/img_res_fit_polynomial.jpg',out_img)
    #plt.imshow(out_img) # Uncomment to see polynomial fit overlay on image

    return out_img, left_polyfit, right_polyfit, center_offset_pixels  # polynomial lines x & y values

def fit_polynomial_plots(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    poly_left = np.polyfit(lefty,leftx,2)
    poly_right = np.polyfit(righty,rightx,2)
    # Generate x and y values for plotting
    # Generate y values to feed into polynomials
    # from y=0, to y=719, # of values: 720
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = poly_left[0]*ploty**2 + poly_left[1]*ploty + poly_left[2]
    right_fitx = poly_right[0]*ploty**2 + poly_right[1]*ploty + poly_right[2]
    
    return left_fitx, right_fitx, ploty, poly_left, poly_right   

# %% Search Around Lane-Lines
def search_lanes_around_poly(img_binary_warped,poly_left_fit,poly_right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50

    # Grab activated pixels
    nonzero = img_binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox >= (poly_left_fit[0]*nonzeroy**2 + poly_left_fit[1]*nonzeroy + poly_left_fit[2]) - margin) & 
                     (nonzerox < (poly_left_fit[0]*nonzeroy**2 + poly_left_fit[1]*nonzeroy + poly_left_fit[2]) + margin)).nonzero()[0] 
    right_lane_inds = ((nonzerox >= (poly_right_fit[0]*nonzeroy**2 + poly_right_fit[1]*nonzeroy + poly_right_fit[2]) - margin) & 
                     (nonzerox < (poly_right_fit[0]*nonzeroy**2 + poly_right_fit[1]*nonzeroy + poly_right_fit[2]) + margin)).nonzero()[0]    
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, poly_left, poly_right = fit_polynomial_plots(img_binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    img_overlay = np.dstack((img_binary_warped, img_binary_warped, img_binary_warped))*255
    img_window = np.zeros_like(img_overlay)
    # Color in left and right line pixels
    img_overlay[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    img_overlay[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    # Left lane window
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # Right lane window
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(img_window, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(img_window, np.int_([right_line_pts]), (0,255, 0))
    img_search_area = cv2.addWeighted(img_overlay, 1, img_window, 0.3, 0)
    
    # Draw poly fit line on image
    isClosed = False
    right_line_poly = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    left_line_poly = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    img_search_area_poly = cv2.polylines(img_search_area,np.int32([right_line_poly]),isClosed,(255,255,0),thickness=2)
    img_search_area_poly = cv2.polylines(img_search_area_poly,np.int32([left_line_poly]),isClosed,(255,255,0),thickness=2)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    # Calculate center offset
    midpoint = int(img_binary_warped.shape[1]/2)
    leftx_base = left_fitx[img_binary_warped.shape[0]-1]
    rightx_base = right_fitx[img_binary_warped.shape[0]-1]
    # Calculate vehicle offset from lane center (in pixels)
    center_offset_pixels = ((rightx_base-leftx_base)-midpoint) #np.absolute 
    



    return img_search_area_poly, left_fitx, right_fitx, ploty, poly_left, poly_right, center_offset_pixels 

# %% Calculate Lane Curvature
def calculate_lane_curvature(y_pos,left_polyfit,right_polyfit,ym_per_pix,xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in radians.
    '''
        
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    #print(ploty,left_fit_cr,right_fit_cr)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = y_pos*ym_per_pix
    
    #scale_1 = xm_per_pix/ym_per_pix
    #scale_2 = xm_per_pix/ym_per_pix**2
    scaling_coeff = [xm_per_pix/ym_per_pix,xm_per_pix/ym_per_pix**2]
    #print ('y_eval: ',y_eval,'scaling_coeff:',scaling_coeff)
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*scaling_coeff[1]*left_polyfit[0]*y_eval+scaling_coeff[0]*left_polyfit[1])**2)**(3/2))/np.absolute(2*scaling_coeff[1]*left_polyfit[0])  ## Implement the calculation of the left line here
    right_curverad = ((1+(2*scaling_coeff[1]*right_polyfit[0]*y_eval+scaling_coeff[0]*right_polyfit[1])**2)**(3/2))/np.absolute(2*scaling_coeff[1]*right_polyfit[0])  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

# %% Overlay Calculations and De-Warp image

def overlay_result(img_original, img_warped_binary, left_polyfit, right_polyfit, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    pts_ploty = np.linspace(0, img_warped_binary.shape[0]-1, img_warped_binary.shape[0])
    pts_left_fitx = left_polyfit[0]*pts_ploty**2 + left_polyfit[1]*pts_ploty + left_polyfit[2]
    pts_right_fitx = right_polyfit[0]*pts_ploty**2 + right_polyfit[1]*pts_ploty + right_polyfit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([pts_left_fitx, pts_ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([pts_right_fitx, pts_ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_rewarp = cv2.warpPerspective(color_warp, Minv, (img_original.shape[1], img_original.shape[0])) 
    # Combine the result with the original image
    img_res_overlay = cv2.addWeighted(img_original, 1, img_rewarp, 0.3, 0)
    # 
    return img_res_overlay

    plt.imshow(result)

 # %% Project Steps - Image Analysis
def lanefindingpipeline(selected_test_image):
    # Load Image
    img_tst = cv2.imread('../test_images/' + selected_test_image)
    #0) Project / Variable Setup
    points_object = np.float32([[200,720],[590,450],[700,450],[1075,720]])
    points_image = np.float32([[300,720],[300,0],[975,0],[975,720]])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension - 0.04166
    xm_per_pix = 3.7/700 # meters per pixel in x dimension - 0.005285

    #1) Camera calibration - Compute Transformation between 3D points in the world vs 2D image points
    # Run cameracalibration_gen() once to create Pickle file with Camera Matrix and Distortion Coefficients


    #2) Distortion correction - Ensure geometrical shape of object is represented consistently troughout image
    image_undistort = cameracalibration_undistort(img_tst)

    #3) Color/gradient threshold
    img_res_colorgrad = colorgradthreshold(image_undistort)


    #4) Perspective transform
    img_tst_perstrans = img_res_colorgrad
    img_res_perstrans, matrix_trans, matrix_inv_trans = \
        perspectivetransform(img_tst_perstrans,points_object,points_image)

    #5) Detect lane lines
    histogram = np.sum(img_res_perstrans[img_res_perstrans.shape[0]//2:,:], axis=0)
    plt.figure()
    plt.plot(histogram)
    # Detect Lane Lines - Sliding Window
    img_res_lanelines, poly_lane_left, poly_lane_right, center_offset_pixels = search_lanes_sliding_win(img_res_perstrans)
    # Detect Lane Lines - Polynomial-Fit Guided Search
    img_res_lanelines_search_area, points_left_fitx, points_right_fitx, points_ploty, poly_left, poly_right, center_offset_pixels = \
        search_lanes_around_poly(img_res_perstrans,poly_lane_left,poly_lane_right)

    #6) Calculate lane curvature (in real-space, not pixel-space)
    left_curverad,right_curverad = calculate_lane_curvature(img_res_perstrans.shape[0],poly_lane_left,poly_lane_right,ym_per_pix,xm_per_pix)
    print('left_curverad: ',left_curverad,' meters \r\n','right_curverad: ',right_curverad, 'meters \r\n') #,poly_lane_left,poly_lane_right)

    #7) Overlay Calculations and De-Warp image
    img_res_overlay = overlay_result(img_tst, img_res_colorgrad, poly_lane_left, \
        poly_lane_right, matrix_inv_trans)

    #8) Overlay Radius Calculations - Make this rolling average of 5? 10? measurements
    # Calculate Average Radius
    avg_curverad = (left_curverad + right_curverad)/2
    # Calculate Vechicle Offset from Center Lane
    center_offset = center_offset_pixels * xm_per_pix
    # Generate Text Overlay - Vechicle Offset
    text_overlay_os = "Vehicle Offset from Lane Center = %.2f (meters) "%center_offset
    # Generate Text Overlay - Land Radious
    if (avg_curverad <= 2000):
        text_overlay_rad = "Lane Radius of Curvature = %.0f (meters) " %avg_curverad
    else:
        text_overlay_rad = "Lane Radius of Curvature = N/A (Straight Line - Rad > 2000 (meters)) "
    # Set Text Overlay Parameters
    text_org_1 = (50,50)
    text_org_2 = (50,100)
    text_fontscale = 1
    text_color = (0,0,0)
    text_fontface = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 2
    # Write Text Overlay on Video Images
    cv2.putText(img_res_overlay,text_overlay_rad,text_org_1,text_fontface,text_fontscale,text_color,text_thickness)
    cv2.putText(img_res_overlay,text_overlay_os,text_org_2,text_fontface,text_fontscale,text_color,text_thickness)

    #9) Output Plots!

    # Polt: Undistorted Image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_tst)
    ax1.set_title('Original Image: (%s)' %(selected_test_image), fontsize=40)
    ax2.imshow(image_undistort)
    ax2.set_title('Undistorted Image: Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    cv2.imwrite('../output_images/img_res_undistort_%s' %(selected_test_image),image_undistort)    
    # Plot: Color/gradient threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image_undistort)
    ax1.set_title('Color Grad Thresh: (%s)' %(selected_test_image), fontsize=40)
    ax2.imshow(img_res_colorgrad)
    ax2.set_title('Color Grad Thresh: Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    cv2.imwrite('../output_images/img_res_colorgrad_%s' %(selected_test_image),img_res_colorgrad)
    # Plot: Perspective Transform
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_tst_perstrans)
    ax1.set_title('Perspective Trans: (%s)' %(selected_test_image), fontsize=40)
    ax2.imshow(img_res_perstrans)
    ax2.set_title('Perspective Trans: Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    cv2.imwrite('../output_images/img_res_perstrans_%s' %(selected_test_image),img_res_perstrans)
    # Plot: Lane Line
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_res_perstrans)
    ax1.set_title('Lane Lines: (%s)' %(selected_test_image), fontsize=40)
    ax2.imshow(img_res_lanelines)
    ax2.set_title('Lane Lines: Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    cv2.imwrite('../output_images/img_res_lanelines_%s' %(selected_test_image),img_res_lanelines)
    # Plot: Lane Line Search Area
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_res_perstrans)
    ax1.set_title('Lane Line Search Area: (%s)' %(selected_test_image), fontsize=40)
    ax2.imshow(img_res_lanelines_search_area)
    ax2.set_title('Lane Line Search Area: Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    cv2.imwrite('../output_images/img_res_searcharea_%s' %(selected_test_image),img_res_lanelines_search_area)
    # Plot: Final Lane Detected Overlay
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_tst)
    ax1.set_title('Overlay Calc and De-Warp: (%s)' %(selected_test_image), fontsize=40)
    ax2.imshow(img_res_overlay)
    ax2.set_title('Overlay Calc and De-Warp: Pipeline ', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    cv2.imwrite('../output_images/img_res_overlay_%s' %(selected_test_image),img_res_overlay)


#%% Define a Circular Ring Buffer for Polynomials
class RingBuffer_Poly():
    """ class that implements a not-yet-full buffer of polynomial constants """
    def __init__(self,size_max):
        self.max = size_max
        #self.data = np.empty([3,0],dtype='float')
        #self.data = np.empty([3,size_max],dtype='float')
        self.data_array = []
        self.data = [] #([],dtype='float')
        self.data.append([])
        self.data.append([])
        self.data.append([])
        self.index = 0
        self.poly_avg = 0

    class __Full:
        """ class that implements a full buffer of polynomial constants """
        def append(self, poly):
            """ Append an element overwriting the oldest one. """
            self.data[0,self.index] = poly[0]
            self.data[1,self.index] = poly[1]
            self.data[2,self.index] = poly[2]
            self.index = (self.index+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            #return self.data[self.index:]+self.data[:self.index]
            return self.data
        def get_avg(self):
            """ return average of element list """
            self.poly_avg = ([np.average(self.data[0,:]),np.average(self.data[1,:]),np.average(self.data[2,:])])
            return self.poly_avg
        def get_isFull(self):
            """ return if element list is full """
            return True

    def append(self,poly):
        """append an element at the end of the buffer"""
        #self.data[0,self.index] = poly[0]
        #self.data[1,self.index] = poly[1]
        #self.data[2,self.index] = poly[2]
        self.data[0].append(poly[0])
        self.data[1].append(poly[1])
        self.data[2].append(poly[2])
        self.index = (self.index+1) % self.max
        #if self.data.shape[1] == self.max:
        if len(self.data[0]) == self.max:
            self.index = 0
            # Convert list to np.array
            self.data = np.array(self.data)
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data
    
    def get_avg(self):
        """ return average of element list """
        self.data_array = np.array(self.data)
        self.poly_avg = ([np.average(self.data_array[0,:]),np.average(self.data_array[1,:]),np.average(self.data_array[2,:])])
        return self.poly_avg

    def get_isFull(self):
        """ return if element list is full """
        return False



#%% Define a Circular Ring Buffer
class RingBuffer():
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []
        self.index = 0

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.index] = x
            self.index = (self.index+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.index:]+self.data[:self.index]
        def get_avg(self):
            """ return average of element list """
            return np.average(self.data)
        def get_isFull(self):
            """ return if element list is full """
            return True

    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.index = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data
    
    def get_avg(self):
        """ return average of element list """
        return np.average(self.data)

    def get_isFull(self):
        """ return if element list is full """
        return False


# %% Define a class to save parameters of each line detection
class Line(RingBuffer_Poly):
    def __init__(self,size_max):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        # x Ring Buffer Length
        self.ringbuff_size = size_max 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.current_poly_avg = [np.array([False])]  
        #polynomial coefficients for the most recent fit
        self.current_poly_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
        #test
        self.test = 0
        super().__init__(size_max)


# %% Project Steps - Video Analysis
def lanefindingpipeline_video(selected_test_image):
    # Create "Line" Class Instances to Store Data
    # lanelinedata_right = Line()
    # lanelinedata_left = Line()
    # Load Image
    img_tst = selected_test_image
    #img_tst = cv2.imread('../test_images/' + selected_test_image)
    #0) Project / Variable Setup
    points_object = np.float32([[200,720],[590,450],[700,450],[1075,720]])
    points_image = np.float32([[300,720],[300,0],[975,0],[975,720]])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension - 0.04166
    xm_per_pix = 3.7/700 # meters per pixel in x dimension - 0.005285

    #1) Camera calibration - Compute Transformation between 3D points in the world vs 2D image points


    #2) Distortion correction - Ensure geometrical shape of object is represented consistently troughout image

    #3) Color/gradient threshold
    img_res_colorgrad = colorgradthreshold(img_tst)


    #4) Perspective transform
    img_tst_perstrans = img_res_colorgrad
    img_res_perstrans, matrix_trans, matrix_inv_trans = \
        perspectivetransform(img_tst_perstrans,points_object,points_image)

    #5) Detect lane lines
    #histogram = np.sum(img_res_perstrans[img_res_perstrans.shape[0]//2:,:], axis=0)
    #plt.figure()
    #plt.plot(histogram)
    # Search for lane lines depending on results from previous frame - Look Ahead Filter
    print(lanelinedata_right.detected)

    # Currently using averaged poly fit constants to perform guided search
    if (lanelinedata_right.detected == True & lanelinedata_left.detected == True):
        # Detect Lane Lines - Polynomial-Fit Guided Search ###RDK POLY FITS HAVE TO BE UPDATED!!
        print("Polynomial-Fit Guided Search")
        img_res_lanelines_search_area, points_left_fitx, points_right_fitx, points_ploty, \
            lanelinedata_left.current_poly_fit, lanelinedata_right.current_poly_fit, center_offset_pixels  = \
            search_lanes_around_poly(img_res_perstrans,lanelinedata_left.current_poly_avg, \
            lanelinedata_right.current_poly_avg)
        if (not(points_left_fitx.any() and points_right_fitx.any())):
            lanelinedata_right.detected = False
            lanelinedata_left.detected = False
    else:
        # Detect Lane Lines - Sliding Window
        print("Sliding Window Search")
        img_res_lanelines, lanelinedata_left.current_poly_fit, lanelinedata_right.current_poly_fit, center_offset_pixels = \
        search_lanes_sliding_win(img_res_perstrans)
        lanelinedata_right.detected = True
        lanelinedata_left.detected = True

    lanelinedata_left.append(lanelinedata_left.current_poly_fit)
    lanelinedata_right.append(lanelinedata_right.current_poly_fit)
    lanelinedata_left.current_poly_avg = lanelinedata_left.get_avg()
    lanelinedata_right.current_poly_avg = lanelinedata_right.get_avg()
    # Use Average Polynomial Constants for further calculations / overlay
    poly_lane_left = lanelinedata_left.current_poly_avg
    poly_lane_right = lanelinedata_right.current_poly_avg
    # To Do AVG : Append poly lane data to class
        
    #6) Calculate lane curvature (in real-space, not pixel-space)
    left_curverad,right_curverad = calculate_lane_curvature(img_res_perstrans.shape[0],poly_lane_left,poly_lane_right,ym_per_pix,xm_per_pix)
    #print('left_curverad: ',left_curverad,' meters \r\n','right_curverad: ',right_curverad, 'meters \r\n') #,poly_lane_left,poly_lane_right)

    #7) Record Lane Data
    lanelinedata_right.radius_of_curvature = right_curverad
    lanelinedata_left.radius_of_curvature = left_curverad


    #7) Overlay Calculations and De-Warp image
    img_res_overlay = overlay_result(img_tst, img_res_colorgrad, poly_lane_left, \
        poly_lane_right, matrix_inv_trans)

    #8) Check
    #lanelinedata_right.test = lanelinedata_right.test +1
    #print("LaneLineData!!",lanelinedata_right.test)
    #print("Right Current: ",lanelinedata_right.current_poly_fit)
    #print("Left Current: ",lanelinedata_left.current_poly_fit)
    #print("Right Current Avg: ",lanelinedata_right.current_poly_avg)
    #print("Left Current Avg: ",lanelinedata_left.current_poly_avg)
    #print("Right List: ",lanelinedata_right.get())
    #print("Left List: ",lanelinedata_left.get())
    #print("Buffer Full: ", lanelinedata_right.get_isFull())


    #test_ring.append(lanelinedata_right.test)
    #
 
    #print(lanelinedata_right.test,lanelinedata_right.get_avg(),lanelinedata_right.get_isFull())

    #8) Overlay Radius Calculations - Make this rolling average of 5? 10? measurements
    # Calculate Average Radius
    avg_curverad = (left_curverad + right_curverad)/2
    # Calculate Vechicle Offset from Center Lane
    center_offset = center_offset_pixels * xm_per_pix
    # Generate Text Overlay - Vechicle Offset
    text_overlay_os = "Vehicle Offset from Lane Center = %.2f (meters) "%center_offset
    # Generate Text Overlay - Land Radious
    if (avg_curverad <= 2000):
        text_overlay_rad = "Lane Radius of Curvature = %.0f (meters) " %avg_curverad
    else:
        text_overlay_rad = "Lane Radius of Curvature = N/A (Straight Line - Rad > 2000 (meters)) "
    # Set Text Overlay Parameters
    text_org_1 = (50,50)
    text_org_2 = (50,100)
    text_fontscale = 1
    text_color = (0,0,0)
    text_fontface = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 2
    # Write Text Overlay on Video Images
    cv2.putText(img_res_overlay,text_overlay_rad,text_org_1,text_fontface,text_fontscale,text_color,text_thickness)
    cv2.putText(img_res_overlay,text_overlay_os,text_org_2,text_fontface,text_fontscale,text_color,text_thickness)
    
    return img_res_overlay

# %% Project Steps - Video Analysis - Colour Transform
def lanefindingpipeline_perstans_video(selected_test_image):
    # Load Image
    img_tst = selected_test_image
    #img_tst = cv2.imread('../test_images/' + selected_test_image)
    #0) Project / Variable Setup
    points_object = np.float32([[200,720],[590,450],[700,450],[1075,720]])
    points_image = np.float32([[300,720],[300,0],[975,0],[975,720]])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension - 0.04166
    xm_per_pix = 3.7/700 # meters per pixel in x dimension - 0.005285

    #1) Camera calibration - Compute Transformation between 3D points in the world vs 2D image points


    #2) Distortion correction - Ensure geometrical shape of object is represented consistently troughout image

    #3) Color/gradient threshold
    img_res_colorgrad = colorgradthreshold(img_tst)


    #4) Perspective transform
    img_tst_perstrans = img_res_colorgrad
    img_res_perstrans, matrix_trans, matrix_inv_trans = \
        perspectivetransform(img_tst_perstrans,points_object,points_image)

    img_res_perstrans_reshape = np.expand_dims(img_res_perstrans,axis=2)
    img_res_perstrans_reshape = np.dstack((img_res_perstrans_reshape,img_res_perstrans_reshape,img_res_perstrans_reshape))

    return img_res_perstrans_reshape

# %% Project Steps - Video Analysis - Perspective Transform
def lanefindingpipeline_colourtans_video(selected_test_image):
    # Load Image
    img_tst = selected_test_image
    #img_tst = cv2.imread('../test_images/' + selected_test_image)
    #0) Project / Variable Setup
    points_object = np.float32([[200,720],[590,450],[700,450],[1075,720]])
    points_image = np.float32([[300,720],[300,0],[975,0],[975,720]])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension - 0.04166
    xm_per_pix = 3.7/700 # meters per pixel in x dimension - 0.005285

    #1) Camera calibration - Compute Transformation between 3D points in the world vs 2D image points


    #2) Distortion correction - Ensure geometrical shape of object is represented consistently troughout image

    #3) Color/gradient threshold
    img_res_colorgrad = colorgradthreshold(img_tst)
    img_res_colorgrad_reshape = np.expand_dims(img_res_colorgrad,axis=2)
    img_res_colorgrad_reshape = np.dstack((img_res_colorgrad_reshape,img_res_colorgrad_reshape,img_res_colorgrad_reshape))

    #print(img_res_colorgrad.shape)
    #img_res_colorgrad = img_res_colorgrad
    return img_res_colorgrad_reshape

# %% Project Steps - Video Analysis - LaneLine Detection
def lanefindingpipeline_lanelines_video(selected_test_image):
    # Load Image
    img_tst = selected_test_image
    #img_tst = cv2.imread('../test_images/' + selected_test_image)
    #0) Project / Variable Setup
    points_object = np.float32([[200,720],[590,450],[700,450],[1075,720]])
    points_image = np.float32([[300,720],[300,0],[975,0],[975,720]])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension - 0.04166
    xm_per_pix = 3.7/700 # meters per pixel in x dimension - 0.005285

    #1) Camera calibration - Compute Transformation between 3D points in the world vs 2D image points


    #2) Distortion correction - Ensure geometrical shape of object is represented consistently troughout image

    #3) Color/gradient threshold
    img_res_colorgrad = colorgradthreshold(img_tst)


    #4) Perspective transform
    img_tst_perstrans = img_res_colorgrad
    img_res_perstrans, matrix_trans, matrix_inv_trans = \
        perspectivetransform(img_tst_perstrans,points_object,points_image)

    #5) Detect lane lines
    #histogram = np.sum(img_res_perstrans[img_res_perstrans.shape[0]//2:,:], axis=0)
    #plt.figure()
    #plt.plot(histogram)
    # Search for lane lines depending on results from previous frame

    # Detect Lane Lines - Sliding Window
    #print("Sliding Window Search")
    img_res_lanelines, lanelinedata_left_video.current_poly_fit, lanelinedata_right_video.current_poly_fit, center_offset_pixels = \
    search_lanes_sliding_win(img_res_perstrans)
    #lanelinedata_right_video.detected = True
    #lanelinedata_left_video.detected = True

    #if (lanelinedata_right_video.detected == True & lanelinedata_left_video.detected == True):
    #    # Detect Lane Lines - Polynomial-Fit Guided Search ###RDK POLY FITS HAVE TO BE UPDATED!!
    #    print("Polynomial-Fit Guided Search")
    #    img_res_lanelines, points_left_fitx, points_right_fitx, points_ploty, \
    #        lanelinedata_left_video.current_poly_fit, lanelinedata_right_video.current_poly_fit  = \
    #        search_lanes_around_poly(img_res_perstrans_search,lanelinedata_left_video.current_poly_fit, \
    #        lanelinedata_right_video.current_poly_fit)
    #    if (not(points_left_fitx.any() and points_right_fitx.any())):
    #        lanelinedata_right_video.detected = False
    #        lanelinedata_left_video.detected = False
    #else:
    #    # Detect Lane Lines - Sliding Window
    #    print("Sliding Window Search")
    #    img_res_lanelines, lanelinedata_left_video.current_poly_fit, lanelinedata_right_video.current_poly_fit, center_offset_pixels = \
    #    search_lanes_sliding_win(img_res_perstrans)
    #    lanelinedata_right_video.detected = True
    #    lanelinedata_left_video.detected = True

    #print(img_res_lanelines.shape)
    #img_res_lanelines_reshape = np.expand_dims(img_res_lanelines,axis=2)
    #img_res_lanelines_reshape = np.dstack((img_res_lanelines_reshape,img_res_lanelines_reshape,img_res_lanelines_reshape))

  
    return img_res_lanelines

# %% Project Steps - Video Analysis - LaneLine Detection
def lanefindingpipeline_lanelines_search_video(selected_test_image):
    # Load Image
    img_tst = selected_test_image
    #img_tst = cv2.imread('../test_images/' + selected_test_image)
    #0) Project / Variable Setup
    points_object = np.float32([[200,720],[590,450],[700,450],[1075,720]])
    points_image = np.float32([[300,720],[300,0],[975,0],[975,720]])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension - 0.04166
    xm_per_pix = 3.7/700 # meters per pixel in x dimension - 0.005285

    #1) Camera calibration - Compute Transformation between 3D points in the world vs 2D image points


    #2) Distortion correction - Ensure geometrical shape of object is represented consistently troughout image

    #3) Color/gradient threshold
    img_res_colorgrad = colorgradthreshold(img_tst)


    #4) Perspective transform
    img_tst_perstrans = img_res_colorgrad
    img_res_perstrans, matrix_trans, matrix_inv_trans = \
        perspectivetransform(img_tst_perstrans,points_object,points_image)

    #5) Detect lane lines
    #histogram = np.sum(img_res_perstrans[img_res_perstrans.shape[0]//2:,:], axis=0)
    #plt.figure()
    #plt.plot(histogram)
    # Search for lane lines depending on results from previous frame

    # Detect Lane Lines - Sliding Window
    #print("Sliding Window Search")
    #img_res_lanelines, lanelinedata_left_video.current_poly_fit, lanelinedata_right_video.current_poly_fit, center_offset_pixels = \
    #search_lanes_sliding_win(img_res_perstrans)
    #lanelinedata_right_video.detected = True
    #lanelinedata_left_video.detected = True

    img_res_lanelines_searcharea = img_res_perstrans
    if (lanelinedata_right_video.detected == True & lanelinedata_left_video.detected == True):
        # Detect Lane Lines - Polynomial-Fit Guided Search ###RDK POLY FITS HAVE TO BE UPDATED!!
        #print("Polynomial-Fit Guided Search")
        img_res_lanelines_searcharea, points_left_fitx, points_right_fitx, points_ploty, \
            lanelinedata_left_video.current_poly_fit, lanelinedata_right_video.current_poly_fit, center_offset_pixels  = \
            search_lanes_around_poly(img_res_perstrans,lanelinedata_left_video.current_poly_fit, \
            lanelinedata_right_video.current_poly_fit)
        if (not(points_left_fitx.any() and points_right_fitx.any())):
            lanelinedata_right_video.detected = False
            lanelinedata_left_video.detected = False
    else:
        # Detect Lane Lines - Sliding Window
        #print("Sliding Window Search")
        img_res_lanelines, lanelinedata_left_video.current_poly_fit, lanelinedata_right_video.current_poly_fit, center_offset_pixels = \
        search_lanes_sliding_win(img_res_perstrans)
        lanelinedata_right_video.detected = True
        lanelinedata_left_video.detected = True

    #print(img_res_lanelines.shape)
    #img_res_lanelines_reshape = np.expand_dims(img_res_lanelines,axis=2)
    #img_res_lanelines_reshape = np.dstack((img_res_lanelines_reshape,img_res_lanelines_reshape,img_res_lanelines_reshape))

  
    return img_res_lanelines_searcharea

# %% Run Image Analysis Loop  #################################################################
# Get Image Bank
image_bank = os.listdir("../test_images/")
#print ("This is %s" %(image_bank[0])) # Test Dynamic Naming
for selected_test_image in image_bank:
   print ("Processing Image: %s" %(selected_test_image)) # Test Dynamic Naming
   #print ("%s" %(selected_test_image))
   # Run Pipeline
   lanefindingpipeline(selected_test_image)

# %% Run Video Analysis Loop ##################################################################
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# Create "Line" Class Instances to Store Data
lanelinedata_right = Line(5)
lanelinedata_left = Line(5)
lanelinedata_right.detected = False
lanelinedata_left.detected = False

#test_ring = RingBuffer(5)
overlay_output = '../output_video/project_video_output.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("../project_video.mp4").subclip(0,2)
clip1 = VideoFileClip("../project_video.mp4")
overlay_clip = clip1.fl_image(lanefindingpipeline_video) #NOTE: this function expects color images!!
%time overlay_clip.write_videofile(overlay_output, audio=False)

# %% Run Video Analysis Loop - Colour Transform
# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML

overlay_output_colourtrans = '../output_video/project_colourtrans_video_ouput.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
#clip1_ct = VideoFileClip("../project_video.mp4").subclip(0,1)
clip1_ct = VideoFileClip("../project_video.mp4")
overlay_clip_ct = clip1_ct.fl_image(lanefindingpipeline_colourtans_video) #NOTE: this function expects color images!!
%time overlay_clip_ct.write_videofile(overlay_output_colourtrans, audio=False)

# %% Run Video Analysis Loop - Perspective Transform
# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML

overlay_output_perstrans = '../output_video/project_perstrans_video_ouput.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("../project_video.mp4").subclip(0,2)
clip1_pt = VideoFileClip("../project_video.mp4")
overlay_clip_pt = clip1_pt.fl_image(lanefindingpipeline_perstans_video) #NOTE: this function expects color images!!
%time overlay_clip_pt.write_videofile(overlay_output_perstrans, audio=False)

# %% Run Video Analysis Loop - LaneLine Transform
# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
# Create "Line" Class Instances to Store Data
lanelinedata_right_video = Line(5)
lanelinedata_left_video = Line(5)
lanelinedata_right_video.detected = False
lanelinedata_left_video.detected = False

overlay_output_lanelines = '../output_video/project_lanelines_video_ouput.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
#clip1_ll = VideoFileClip("../project_video.mp4").subclip(0,2)
clip1_ll = VideoFileClip("../project_video.mp4")
overlay_clip_ll = clip1_ll.fl_image(lanefindingpipeline_lanelines_video) #NOTE: this function expects color images!!
%time overlay_clip_ll.write_videofile(overlay_output_lanelines, audio=False)

# %% Run Video Analysis Loop - LaneLine Search Area
# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
# Create "Line" Class Instances to Store Data
lanelinedata_right_video = Line(5)
lanelinedata_left_video = Line(5)
lanelinedata_right_video.detected = False
lanelinedata_left_video.detected = False

overlay_output_lanelines_search = '../output_video/project_lanelines_search_video_ouput.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
#clip1_llsa = VideoFileClip("../project_video.mp4").subclip(0,2)
clip1_llsa = VideoFileClip("../project_video.mp4")
overlay_clip_llsa = clip1_llsa.fl_image(lanefindingpipeline_lanelines_search_video) #NOTE: this function expects color images!!
%time overlay_clip_llsa.write_videofile(overlay_output_lanelines_search, audio=False)

# %% Final Video Image Stacking - For Debugging and Reporting Purposes
from moviepy.editor import VideoFileClip, clips_array, vfx
clip1 = VideoFileClip("../project_video.mp4").margin(10) # add 10px contour
clip2 = VideoFileClip("../output_video/project_colourtrans_video_ouput.mp4").margin(10) # add 10px contour
clip3 = VideoFileClip("../output_video/project_perstrans_video_ouput.mp4").margin(10) # add 10px contour
clip4 = VideoFileClip("../output_video/project_lanelines_video_ouput.mp4").margin(10) # add 10px contour
clip5 = VideoFileClip("../output_video/project_lanelines_search_video_ouput.mp4").margin(10) # add 10px contour
clip6 = VideoFileClip("../output_video/project_video_output.mp4").margin(10) # add 10px contour

final_clip = clips_array([[clip1, clip6],
                          [clip2, clip3],
                          [clip4,clip5]])
#final_clip.resize(width=480).write_videofile("../project_video_stacked.mp4")
final_clip.write_videofile("../output_video/project_video_output_stacked.mp4")

# %%
