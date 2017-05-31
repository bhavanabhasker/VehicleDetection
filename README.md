## Vehicle detection and tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image3]: ./images/sliding_windows.png
[image4]: ./images/sliding_heat.png
[image5]: ./images/bboxes_and_heat.png
[video1]: ./projectvideo_output.mp4


---

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the Ipython notebook "VehicleDetection.ipynb"   

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


I tried various combinations of parameters such as `LUV` with the same orientation and pixels per cell. 

I finally settled for the color space `RGB` and nd HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`


####  Classifier using SVM 

I trained a linear SVM using sklearn LinearSVC classifier. The training code uses combination of both HOG and color based features. The code for this can be found in the ipython notebook 'VehicleDetection.ipynb'. For every `car` and `notcar` images in the training set, I first applied color conversion. This process was followed by resizing of the training images and extraction of color and hog features. To extract hog features sklearn's  `hog()` function will be used. 

### Sliding Window Search

The code for sliding window search can be found in the same ipython notebook `VehicleDetection.ipynb`. The y axis for the images and video frames was restricted to [400,650]. The y value was selected to ensure that the cars are captured in the area of interest. The window size was chosen to be 64 with an overlap of 50% between the adjacent windows in both horizontal and vertical dimensions. I chose the scaling factor of 1.5 to capture the vehicles correctly. Here is an example image from this step,	


![alt text][image3]

#### Sample images

I searched using HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. HOG subsampling window search was also implemented. To avoid false positives, I used heat maps to draw bounding boxes.  Here are some example images:

![alt text][image4]
---

### Video Implementation

Here's a [link to my video result](./project_video.mp4)


#### False positive detection for overlapping bounding boxes

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

---

### Discussion
Linear SVM classifier is used in the current implementation. It provides good accuracy but consumes some time to process around 16K images. I would like to experiment with different classification algorithms in the next version to ensure scalability and for better prediction. 

Combined hog feature extraction and color based features are used in the project. I have tried feature extraction using color spaces such as HLS, LUV and YCrCb.  Although,LUV initially performed better,'RGB' and 'YCrCb' was selected as it provided better results. The HOG subsampling identified a single bounding box for two vehicles. I would like to try tuning the parameters to make sure that two cars will be identified seperately in the next version.  

