import cv2
import csv
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:    
        lines.append(line)

images = []
measurements = []
for line in lines:
    # load center, left and right image
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
    # correct the left and right image by an offset.
    offset=0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+offset) # left image - add offset
    measurements.append(measurement-offset) # right image 

# Augment training data set by flipping them horizontally.
# This saves me another test drive in the opposite direction
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement) 
    augmented_images.append(cv2.flip(image,1)) # flip around y axis
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
Y_Train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D,  Cropping2D
from keras.layers.pooling import MaxPooling2D

# using NVidias end-to-end approach
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# cropping top and bottom to not use landscape and hood
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 5 convolutional layers with subsampling in the first three layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
# flattening layer 
model.add(Flatten())
# four fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_Train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')