# **Finding Lane Lines on the Road**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
1. Convert image to grayscale - including a gaussian blur
2. Apply canny edge detection
3. Define a mask for limiting the subsequent algorithms to a certain region of the image
4. Hough transformation - find lines among the detected edges
5. Draw a single line for left and right lane based on the output of the Hough transformation

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calcualting the slope and y-intercept of each line. Only certain slope values are allowed to be taken into account for the final left/right lane detection.
After collecting all lines with slopes in a correct range for left/right lane I derive the linear equation from the averaged slope and y-intercept values. A function get_x_values_for_lines() is calculating all needed x-vlaues for given y values for left and right lane.
The y values are given from step 3 (masking) and are derived from all seen lines (y_top & y_bottom)

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when a lane parallel to the others is detected - some weird pattern on the road. Since the current pipeline is optimized for two lanes this could distort the outcome.
Another potential shortcoming would be sharp curves. This needs to be tested with different samples.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to implement a temporal interpolation between the frames of the video sequences. That would stabilize the output for videos.

Another potential improvement could be to automatically adapt the parameters of the pipeline (edge detection parameters & hough transform parameters) based on the current image or video seuqence. But this kind of preprocessing reuqires another pipeline.
