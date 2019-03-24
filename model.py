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

lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader :
        lines.append(line)
        
# images = []
# measurements = []

# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = '/opt/carnd_p3/data/IMG/' + filename
#     image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
#     images.append(image)
#     measurement = float(line[3])
#     measurements.append(measurement)
    
# X_train = np.array(images)
# y_train = np.array(measurements)

training_data, validation_data = train_test_split(lines,test_size=0.15)

def generator(lines, batch_size=32):
    len_data = len(lines)
    while 1: 
        shuffle(lines)
        for offset in range(0, len_data, batch_size):
            batch_data = lines[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_data:
#                     for i in range(0,3):
                        path = '/opt/carnd_p3/data/IMG/'+line[0].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) 
                        center_angle = float(line[3])
                        images.append(center_image)
                        angles.append(center_angle)
                                                
#                         if(i==0):
#                             angles.append(center_angle)
#                         elif(i==1):
#                             angles.append(center_angle+0.2)
#                         elif(i==2):
#                             angles.append(center_angle-0.2)
                        
#                         images.append(cv2.flip(center_image,1))
#                         if(i==0):
#                             angles.append(center_angle*-1)
#                         elif(i==1):
#                             angles.append((center_angle+0.2)*-1)
#                         elif(i==2):
#                             angles.append((center_angle-0.2)*-1)                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
train_generator = generator(training_data, batch_size=32)
validation_generator = generator(validation_data, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5, 5), strides =(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides =(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides =(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, validation_steps=len(validation_data), epochs=1, 
                    validation_data=validation_generator, steps_per_epoch= len(training_data))
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
    