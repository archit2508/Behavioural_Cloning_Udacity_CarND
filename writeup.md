# **Behavioral Cloning Project** 

---

**The goals / steps of this project are the following:**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_for_writeup/1.png "Model Visualization"
[image2]: ./images_for_writeup/2.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing the recording of autonomous driving on the first track using my model
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used a powerful network given by NVIDIA (model.py lines 47-71) suggested in the lectures. This architecture uses a normalization layer at the bottom using a Keras lambda layer followed by 5 convolution layers and 4 fully connected layers after that. I have also cropped the upper sky and trees portion from the images before feeding them into the network.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model didnt show cases of overfitting so I could train my model without including dropout layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 43-44). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used data provided by Udacity and was able to have good results with autonomous driving of the car. The car is keeping on the road in autonomous mode without deviating much towards the edges and recovering itself back to center when deviated. 

![alt text][image2]

### Solution Design Approach

I used the NVIDIA network architecture as suggested in the Udacity lectures. As this architecture has already proven its worth before and was giving good results in my case too, I decided to go with it without any modifications.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on both the training and validation set. This implied that the model was working fine. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when I was feeding whole data at once to train. To improve the driving behavior in these cases, I used generator to train my model in batches.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.