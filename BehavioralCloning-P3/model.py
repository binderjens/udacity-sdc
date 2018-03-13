import cv2
import csv
import numpy as np

# loading the csv data into lines
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:    
        lines.append(line)

#randomize input array
np.random.shuffle(lines)
offset=0.2
offsets=(0,offset,-offset)

def generate_images(batch_size=32):
    while True:
        images = []
        measurements = []
        # randomize the used line
        random_ind = np.random.randint(0, len(lines), batch_size)
        for ind in random_ind:
            line=lines[ind]
            # load center, left or right image based on random number
            i = np.random.randint(0, 3)
            source_path = line[i]
            filename = source_path.split('\\')[-1]
            current_path = './data/IMG/' + filename
            image = cv2.imread(current_path)
            measurement = float(line[3])+offsets[i]
            random_coin = np.random.randint(0,2)
            # flip randomly the image
            if(random_coin==1):
                images.append(cv2.flip(image,1))
                measurements.append(measurement*-1.0)
            else:
                images.append(image)
                measurements.append(measurement)
        yield np.array(images), np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D,  Cropping2D

# using NVidias end-to-end approach
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
# normalize the images with the help of Keras.Lambda
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
# Three fully connected layers
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
# Output layer
model.add(Dense(1))

# compile the model using adam optimizer - no need to tune the learning rate
model.compile(loss='mse', optimizer='adam')
# start learnin - using 20% for validation - use 3 epochs
model.fit_generator(generate_images(),
                    samples_per_epoch=5592,
                    nb_epoch=3,
                    validation_data=generate_images(),
                    nb_val_samples=1398,
                    verbose=1)

model.save('model.h5')