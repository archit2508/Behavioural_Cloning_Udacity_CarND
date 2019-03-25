import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import cv2
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense 
from keras.layers import Lambda, Conv2D, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

lines = []  #Array which will store all data items in csv file
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader :
        lines.append(line)

#85% data will be used for training and 15% for validation
training_data, validation_data = train_test_split(lines,test_size=0.15)

#Generator which will return data items in batches of 32 items
def generator(lines, batch_size=32):
    len_data = len(lines)
    while 1: 
        shuffle(lines)
        for offset in range(0, len_data, batch_size):
            batch_data = lines[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_data:
                path = '/opt/carnd_p3/data/IMG/'+line[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) 
                center_angle = float(line[3])
                images.append(center_image)
                angles.append(center_angle)            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
train_generator = generator(training_data, batch_size=32)
validation_generator = generator(validation_data, batch_size=32)

#NVIDIA Model architecture used to train the model
model = Sequential()
#Preprocessing incoming data using normalization and generalizing small standard deviation for all data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#Cropping the images in dataset to remove sky and trees section to have only area containing road for better training
model.add(Cropping2D(cropping=((70,25),(0,0))))
#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Conv2D(24, (5, 5), strides =(2,2), activation="relu"))
#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Conv2D(36, (5, 5), strides =(2,2), activation="relu"))
#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Conv2D(48, (5, 5), strides =(2,2), activation="relu"))
#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3, 3), activation="relu"))
#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3, 3), activation="relu"))
#flattening image shape into a single array 
model.add(Flatten())
#layer 6- fully connected layer
model.add(Dense(100))
#layer 7- fully connected layer 1
model.add(Dense(50)) 
#layer 8- fully connected layer 1
model.add(Dense(10))
#layer 9- fully connected layer 1
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, validation_steps=len(validation_data), epochs=1, 
                    validation_data=validation_generator, steps_per_epoch= len(training_data))

model.save('model.h5')
    