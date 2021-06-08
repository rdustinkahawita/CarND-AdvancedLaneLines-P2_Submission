
---

# **Advanced Lane Finding Project**

The purpose of this project is to extend the land finding and tracking capability of the pipeline demonstrated in the previous Lane Finding project. To accomplish this, the following steps were taken:

* Computing of the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Applying a distortion correction to raw images.
* Using color transforms, gradients, etc., to create a thresholded binary image.
* Applying a perspective transform to rectify binary image ("birds-eye view").
* Detecting lane pixels and fit to find the lane boundary.
* Determining the curvature of the lane and vehicle position with respect to center.
* Warping the detected lane boundaries back onto the original image.
* Outputting the visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/img_res_calibration1.jpg "Undistorted"
[image2]: ./output_images/img_res_undistort_test4.jpg "Road Transformed"
[image3]: ./output_images/img_res_colorgrad_test4.jpg "Binary Example"
[image4]: ./output_images/img_res_perstrans_test4.jpg "Warp Example"
[image5]: ./output_images/img_res_lanelines_test4.jpg "Fit Visual 1"
[image6]: ./output_images/img_res_searcharea_test4.jpg "Fit Visual 2"
[image7]: ./output_images/img_res_overlay_test4.jpg "Output"
[image8]: ./output_images/img_res_stacked_output.jpg "Stacked Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All code for the camera calibration is contained within the Python file entitled `Udacity_SDC_AdvancedLaneFinding.py`. 

The code for the camera calibration consists of two functions: 
* cameracalibration_gen() -> **line #12**
* cameracalibration_undistort() -> **line #66**

**Note: The `Camera Calibration Initialize and Test` code block (line #92) must be run first before any other code block as it generates the  \"Pickle\" file use within the pipeline to undistort images.**

The `cameracalibration_gen()` function must be called only once as it generates a "Pickle" file with the camera's calibration matrix and distortion coefficients. These values are based off the calibration images found in the  files based on the images found in the  "/camera_cal/" folder. The `cameracalibration_gen()` function works by assuming "object points" are fixed in two dimensions and that these "object points" will be identical for all images. Hence the "object points" are created artificially assuming a specific grid size. The "image points" are determined by using OpenCV's `cv2.findChessboardCorners()` function. For every image where the `cv2.findChessboardCorners()` function identifies the chessboard corners, the corner values ("image points") along with the artificially created "object points" are appended to an array. This array is then used via OpenCV's `cv2.calibrateCamera()` function to to calculate the camera's calibration matrix and distortion coefficients which are subsequently stored in a "Pickle" file. 

The `cameracalibration_undistort()` function extracts the camera's calibration matrix and distortion coefficients, previously stored in a "Pickle" file, and generates the undistorted using the `cv2.undistort()` function. 

This distortion correction when applied to the original test image yields the following result: 

![alt text][image1]

### Pipeline (single images)

All code for the "single images" pipeline are contained within the Python file entitled `Udacity_SDC_AdvancedLaneFinding.py` file. The pipeline for single images is called via from **line #1073** via the code block indicated as `Run Image Analysis Loop`. The code creates a list of all images found in the "/test_images/" directory and then funnels each image into the pipeline via the function `lanefindingpipeline(selected_test_image)` on **line #500**. 

Note: The "Test 4" image was selected to highlight the constructed pipeline as it proved to be one of the more challenging given that includes a transition from dark to light asphalt. 

#### 1. Provide an example of a distortion-corrected image.

Using the aforementioned `cameracalibration_undistort()` function as the first step of the pipeline, the following undistorted image was generated from the original "test 4" image:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Using the `colorgradthreshold()` function on **line #111**, a binary image is generated based of the undistorted image generated in the previous step. 

The `colorgradthreshold()` function carries out color thresholding on the saturation channel of the image. The saturation channel was selected as it yielded the best representation of the lane lines against both light and dark asphalt. The saturation channel was "binarized" by setting pixels that were within the threshold limits to a boolean vale of "1" while pixels outside of the limits were kept the default value of "0".

The `colorgradthreshold()` function also used a gradient threshold to eliminate noise within the image. Based on the fact that the lane lines appear mostly vertical an image, the Sobel operator was employed. Taking the gradient in the x direction better emphasized the edges closer to vertical and hence was best for the detection of lane lines. Next, the absolute value of the derivative was calculated before scaling the values to generate an 8-bit image. Finally the Sobel gradient was "binarized" by setting pixels that were within the threshold limits to a boolean vale of "1" while pixels outside of the limits were kept the default value of "0".

The threshold values for both the S-Channel and Sobel x-gradient are displayed in the table below:

| S-Channel     | Sobel X-Grad  | 
|:-------------:|:-------------:| 
| Minimum:      | Minimum:      | 
| 170           | 20            |
| Maximum:      | Maximum:      |
| 255           | 100           |

Finally, the "binarized" saturation channel and Sobel x-gradient images are stacked and only pixels set to "1" in both the saturation and Sobel images are kept. The resulting array is converted into an 8-bit image, an example of which is displayed below:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Using the `perspectivetransform()` function on **line #187**, a perspective transform is carried out on the binarized color and gradient thresholded image generated in the previous step.

Within the `perspectivetransform()` function, a transform matrix is calculated using the OpenCV function `cv2.getPerspectiveTransform()`. This function calculates the transform matrix from inputted source and destination points. The transform matrix is then subsequently used by the OpenCV function `cv2.warpPerspective()` to generate the warped image yielding a "birds-eye-view" of the lane lines as demonstrated in the image below. As expected, the lane lines appear to be relatively parallel.

![alt text][image4]

The source and destination points used to generate the transform matrix by the `cv2.getPerspectiveTransform()` were selected by manual inspection of the test images. The source and destination points used are presented in the table below:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 300, 0        | 
| 200, 720      | 300, 720      |
| 1075, 720     | 975, 720      |
| 700, 450      | 975, 0        |

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Identification of lane lines and determining their corresponding best-fit polynomial is carried out via the `search_lanes_sliding_win()` function on **line # 315**.

In order to identify lane lines in the binarized "birds eye view" image presented above, first the lane pixels were identified via the `find_lane_pixels()` function on **line #207** (called from within the `search_lanes_sliding_win()` function). First a histogram of the binarized image is taken. Using the NumPy's `np.argmax()` function on the histogram yields the highest value along x position of the image. This consistently works out as a reliable method for detecting the start of the left and right lane lines along the base of the image. 

Next, the "sliding window" method is employed to detected and identify the remaining pixels belonging to the left and right lane lines. Windows are generated within which a minimum number of positive (white) pixels are searched for. Should the minimum number of pixels not be found within the window, the window itself is shifted and the search re-starts. The size of the windows used and the minimum number of pixel required are hyper-parameters set within the function.These hyper-parameters, selected and tuned manually, are presented in the table below.

| Number of Windows | +/- Window Width | Minimum Pixels |
|:-----------------:|:----------------:|:--------------:| 
| 9 Windows         | 75 Pixels        | 50 Pixels      |

The image below shows the identified left and right lane pixels (colored in blue and red respectively) along with a visualization of the sliding window  

![alt text][image5]

With the lane lines pixels identified and returned to the `search_lanes_sliding_win()` function via the `find_lane_pixels()`, a 2nd order polynomial can now be fitted. Using the 'x' and 'y' positions of the lane line pixels, the polynomial coefficients are calculated using NumPy's `np.polyfit()` function. 

The image below shows the polynomial fitted to the left and right lane pixels identified via the `find_lane_pixels()` function.    

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated via the `calculate_lane_curvature()` function on **line #447**. The function takes in polynomial constants for the left and right lane lines (calculated previously via the `find_lane_pixels()` function) and uses the formula below to determine the lane's curvature:

    R_curve = ((1+(2Ay+B)^3)^3/2)/Abs(2A)

 Here "A" and "B" are the coefficients of the 2nd order polynomials used to approximate the lane lines. The "y" value represents the coordinate along the y-axis of the image. To approximate the curvature at the point closest to the vehicle, a "y" value was chosen representing the bottom of the image. 

    f(y) = Ay^2 + By + C

The position of the vehicle offset from lane center is calculated via `find_lane_pixels()` function on **line # 207**. This function is called from the `search_lanes_sliding_win function()` on **line #315**. The center offset (in pixels) is calculated based on the histogram used to detect the starting position of the left and right lane lines. The mid point between the detected left and right lane lines is compared to the vertical centerline of the image to determine the center offset. The offset is then converted from pixels into meters on **line #548**      

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, the detected lane lines are used to create an augmented image via the  `overlay_result()` function on **line #472**. The function takes in the left and right line polynomial constants and generates a colored polygon - representing the area between the detected lane lines - on a blank warped image. Recall the polynomial constants represent lines as seen from the warped "birds-eye" perspective. Hence this image is then "re-warped" back to the perspective of the original image using the inverse transform matrix and the OpenCV function `cv2.warpPerspective()` as seen on **line #491**. Finally the augmented final image is created using the `cv2.addWeighted()` function to overlay the detected area between the lane lines on the original image - as shown below. Note the calculated curvature and centerline offset are also overlaid on the final image.

![alt text][image7]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

All code for the "video" pipeline are contained within the Python file entitled `Udacity_SDC_AdvancedLaneFinding.py` file. The pipeline for video images is called from **line #1083** via the cell indicated as `Run Video Analysis Loop`. The code steps through the video and funnels each image into the pipeline via the function `lanefindingpipeline_video(selected_test_image)` on **line #769**. The output video is saved as `../output_video/project_video_output.mp4`. A link to the output video may be found below. 

The video lane finding pipeline relies on the same basic structure of the image pipeline with a few extra features:
* Sliding Window to Polynomial Line Search Area Hybrid Lane Finding:
    * In order to optimize the lane finding algorithm, instead of carrying out a histogram and using the sliding box method on each frame to detect lane line pixels a hybrid algorithm was implemented. The hybrid algorithm uses the histogram and sliding box method to initially detect the lane lines and generate a second order polynomial to represent each line. In subsequent frames, the search for lane line pixels is carried out in an area surrounding the polynomial line instead of "blindly" searching for lane lines on each frame. This allows for a sort of "feed forward" approach where the search area is intelligently targeted leading to a more efficient lane detection algorithm. Once the new lane line pixels are detected, new polynomial lines are generated for the current frame and the process continues. If no pixels are detected via the targeted searching of the area surrounding the polynomial lines, then the algorithm reverts to the sliding box method. This code is implemented in the `lanefindingpipeline_video(selected_test_image)` function from **lines #804 -> #821**. Note that with the Hybrid Lane Finding algorithm, the calculation of the lane offset is carried out by the active model. It was observed that the lane offset when calculated with the polynomial line search area algorithm was consistently higher than when calculated via the histogram results in sliding window algorithm. 
* Polynomial Averaging: 
    * As one would expect, running the lane finding pipeline on each frame can lead to some "jitter" in the resulting detection. This was also observed in the first assignment and was remedied via a rolling average. The same solution was implemented in this assignment by way of a "ring buffer" for the polynomial coefficients. A user-settable ring buffer was created (size of 5 used in the linked video) via the `RingBuffer_Poly()` class on **line #626** which is inherited by the `Line()` class on **line #739**. Implementing the ring buffer removed much of the original "jitter" and proved not too severe of a "low pass" filter allowing close lane following into and out of curved sections without too much noticeable delay.  

The link to the resulting lane detection video may be found here:  [link to my video result](./output_video/project_video_output.mp4)

#### 2. To better visualize the complete pipeline, I've attached a video which simultaneously demonstrates the various operations carried out via the pipeline. These operations include:
* Original input video
* Colour transform leading to a thresholded binarized output
* Perspective transform leading to a "birds-eye" perspective output
* Lane search algorithms using both the "sliding box" and "search area" method to efficiently detect the and identify lane lines
* Final output video
Stacking the various videos together proved especially useful when debugging instabilities in the pipeline. 

![alt text][image8]

The link to this stacked video output may be found here: [link to my stacked video result](./output_video/project_video_output_stacked.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The image pipeline for lane detection worked relatively well right off the bat. Even the more challenging images with changes in lane line background color seemed to be processed relatively well. Here the critical section of the pipeline ensuring which image treatment to use to create the binarized image. The Sobel operator did a relatively good job of highlighting mostly vertical lines image however the crux of consistent lane detection seemed to rely on the Huge-Saturation-Light breakdown. Using the saturation channel and the correct thresholding values, the lane lines were detected more consistently over varying lighting and back ground conditions.    

The video pipeline initially proved more problematic than its still image counterpart. The main issue being instability in the left lane detection when encountering the change in tarmac colour and hence the contrast of the lane line against the road. This was clearly observed at approximately 22 seconds into the test video. In addition to this, the contrast between the lighter tarmac and the highway divider enters into the binarized image. The combination of these two elements caused sliding windows to jump left momentarily (a bit over a second) before rediscovering the actual left lane line. Rectifying this issue involved a three-fold approach: 
* Immediate improvement was noticed when the size (width-wise) of the sliding box was reduced. This prevented the sliding window to mistaking the highway divider pixels as lane line pixels. This however, on its own, was not a fully robust approach.
* Shifting from a sliding window only approach to a hybrid approach where a search area around the approximated polynomial lines (as described in the previous section) further improved the performance. At this point the lane line detection does not lose track of the left lane line. Although a slight "wobble" may be noticed around the 23 second mark of the resultant video it would not cause a devastating mishap.
* Finally, a ring buffer was added to average out the 2nd degree polynomial lines constants. This approach incrementally improved the pipeline's performance by reducing "jitter" and the previously mentioned "wobble" effect. The ring buffer size is set to a 5-frame average. Increasing the size of the buffer might have completely eliminated the "jtter" and "wobble" issues however this would have had the negative effect of causing the pipeline to be more "sluggish" - a major issues on windy roads with tighter curvature. Therefore a trade off was required and the 5-frame average seemed the best fit.   

Although the performance of the pipeline is adequate for the purpose of this exercise, there are definite improvements that can be integrated to make its performance more robust. These include:
* **Lane Mask** In the previous project, a mask was applied to limit the lane line search area. No such mask was used in this, advanced lane finding, version of the project. However apply a mask to the binarized version of the image could have limited the instability noticed when the tarmac changed colour, as mentioned above. The mask would have prevented the "false positive" of adjacent edges being detected.
* **Colour Gradient Thresholding** Much of the catastrophic instability noticed in the pipeline boiled down to incorrect pixel detection in the binarized version of the input image. Although the Sobel operator and saturation channels were tuned, the resulting image still triggers minor instabilities. Further experimentation with the HSV colour space could be conducted to see if lane detection in the presents of varying tarmac colour, shadows and lighting conditions could be improved. Also, experimentation of applying the Sobel operator to a single HLS or HSV channel may provide more insight into how binarized image can more consistent across the effects and variations mentioned above.
* **Linked / Dynamic Averaging** The ring buffer used to carry out the averaging of the polynomial constants has been set to a size of 5. This was determined through trial and error. A lower number didn't have the desired "dampening" impact on the "jitter" and "wobble". However raising the value to above 5 made the pipeline noticeably sluggish when transitioning from strait to curved road sections and vise versa. Creating a "dynamic" ring buffer where the size of the buffer is automatically adjusted based on the calculation of the curve radius may help improve its robustness. Therefor on long straight sections the right buffer size could be expanded and conversely reduced based on the magnitude of the curve encountered.
* **Additional Sanity Checks** Additional sanity checks could be added to the pipeline in order to reject any anomalies or calculations that are clearly wrong. These sanity checks would likely cost very little  computationally but would prove effective in catching any catastrophic failures of the pipeline. These sanity checks could include, but not limited to:
    * Lane width check: Ensure lane pixels discovered at the bottom, mid and top for the perspective transformed image are within a range for standard highway lane widths
    * Parallel check: Ensure the detected lane lines are parallel to within a given range
    * Check variation of polynomial constants between one frame and the next are within a given range so a to prevent cases where, due to a false detection, the detected lane "jumps"due to a false positive detection
    * etc.           