# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/histogram.png "Visualization"
[image2a]: ./pics/YUV.png "YUV"
[image2b]: ./pics/original.png "original"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./real_world/sign_0.jpg "Traffic Sign 1"
[image5]: ./real_world/sign_1.jpg "Traffic Sign 2"
[image6]: ./real_world/sign_2.jpg "Traffic Sign 3"
[image7]: ./real_world/sign_3.jpg "Traffic Sign 4"
[image8]: ./real_world/sign_4.jpg "Traffic Sign 5"
[image9]: ./real_world/sign_5.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/binderjens/udacity-sdc/tree/master/TrafficSigns-P2/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how often a specific label (x-axis) is present in the training, validation and testing set.

![alt text][image1]

We can see that some of the signs are more represented than others.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images into a different color space - the YUV space. It offers me on the one hand the grayscale image in the Y channel while I do not loose the color information which are still available in U and V channel.

Here is an example of a traffic sign image before and after color conversion.

before: ![alt text][image2b]

after: ![alt text][image2a]

As a last step, I normalized the image data because a data range between -1.0 and +1.0 is much easier to compute.

My approach was to first prepare the original dataset as good as possible in order to train the network to a certain degree. In the histogram can be seen that additional data would be necessary to achieve best results.

After running training, validation and testing of the network it reached already the minimal accepted accuracy of 93% in both validation and testing. I decided to concentrate on tuning the network itself and understand better how tensorflow works rather than spending my time on image tranformation.

Ideally the data set would be enhanced with random modifications of the original images. These modifications includes
* scaling
* rotation
* random noise


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was pretty much using the LeNet5 approach with some minor modifications. It consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 YUV image   							|
| 1st Convolution      	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| 2nd Convolution  	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Flattening            |                                               |
| 1st Fully connected	| Input 400 Output 120  						|
| RELU					|												|
| Dropout				| 0.7 keep probability                          |
| 2nd Fully connected	| Input 120 Output 84    						|
| RELU					|												|
| Dropout				| 0.7 keep probability                          |
| 3nd Fully connected	| Input 84 Output 43    						|
| Softmax				|                           |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer which was also used in the video lectures. I tested randomly other optimizers offered by tensorflow but none of them gave the performance of the Adam optimizer - I'll go deeper into the optimizer topic when time allows.
I used a learning rate of 0.001 and trained the net over 10 epochs with a batch size of 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.946
* test set accuracy of 0.93

If a well known architecture was chosen:
* *What architecture was chosen?*

I chose a LeNet5 architecture known from the video lectures. I applied the grayscaled training set to this network. This gave already a validation accuracy of roughly 90%. The network tend to overfitting - training set accuracy was always around 99% but the net was not able to generalize. I added dropouts in all layers but only the application in the fully connected layers brought sustainable success (validation accuracy of 95%). The increase of the dimension of the first two convolutional layers (6 to 16) speed up the learning time and allowed me to feed also the color channels into the network (and to leave the epoch parameter at 10). These two decisions made the network repeatedly learning to a accuracy of min. 95%.

* *Why did you believe it would be relevant to the traffic sign application?*

Because the domain this network was designed for (recognition of handwritten numbers) is pretty much similar to the traffic signs recognition problem. Only the number of classes is higher and the potential input dimensions.

* *How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?*

By applying the original dataset with relatively few modifications the model achieves always more than 90%. Only small modifications lead to 95%. When I would spend some more time in optimizing the dataset it would easily reach 98% accuracy - as described in the paper mentioned in the notebook.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have choosen six traffic signs from the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]

The first and third image might be difficult to classify because they are distorted patches on the traffic signs.
The second image is covered with snow whereas the other images are easy to recognize.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No passing     		| No passing   									|
| Beware of ice/snow    | Beware of ice/snow    						|
| Children crossing     | Children crossing                             |
| Priority road         | Priority road									|
| Stop   	      		| Stop					 				        |
| Ahead only			| Ahead only      							    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.

I added the plotted probabilities to the html file. It is clear that any sign can be detected in a relatively good probability. Especially sign #2 shows that partly occluded signs can be predicted with a probability of 85%.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

It mainly used the shapes of the sign although from the outputFeatureMap implementation its not clear whether or not any color features would be plotted with the code.
