# **Behavioral Cloning** 

## Writeup Template

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode - this was not changed
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train_model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with five convolutional layers with two different filter sizes (three times 5x5 and two times 3x3) and depths between 24 and 64 (see model.py lines 55-63) 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 51). 

#### 2. Attempts to reduce overfitting in the model 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

In addition the training data was flipped horizontally to allow examples in both directions.

Additional dropout layers were considered but not required for the present result. Nevertheless it may increase the performance of the current model.

In addition the training data was flipped to allow examples in both directions.

Additional dropout layers were not required for a good result but may increase the performance of the current model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. As mentioned already the training data was flipped during pre-processing( line 31-37)

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to apply some of the models mentioned in the lectures. Known models were LeNet, AlexNet and GoogleNet - I created three of them with the help of Keras, trained them and validated the trained model with the simulator.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it is a well known starting point for various classification. As we learned in the lectures it can be used for various other tasks not necessarily image classification.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

I improved the training data by augment the images and normalize them.

The final step was to run the simulator to see how well the car was driving around track one. There were a lot of spots where the vehicle fell off the track and the overall result with the first LeNet approach was rather disappointing.

I kept the pre-processing and applied some other CNNs - including NVidias end-to-end approach (as suggested in the introduction). This directly lead to very good results. Only some cropping was applied and at the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 53-64) consisted of a convolution neural network with the following layers and layer sizes:
- 5 convolutional layers 
- 3 fully connected layers

Here is a visualization of the architecture.
(Taken from https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track #1 using center lane driving. On certain spots I added some recovering data by steering to the side of the road and recovering back to the centerline.

After the collection process the images were flipped which doubled the size of the data set. In the end I had roughly 5500 data points (images + measurements). I then preprocessed this data by cropping the images to only use the actual road and finally normalizing the data. This increased the performance of the model significantly.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation loss was stable on a low level. I used an adam optimizer so that manually training the learning rate wasn't necessary.
