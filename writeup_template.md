# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_arch.png "Model Visualization"
[image2]: ./examples/center_driving.jpg "Center Driving"
[image3]: ./examples/left_recovery.jpg "Recovery Image"
[image4]: ./examples/right_recovery.jpg "Recovery Image"
[image5]: ./examples/crop_image.jpg "Crop Image"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flip_image.jpg "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes below files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Initially I started with simple regression network that converts flattened image into a single output. The input was processed by a Lamda layer that cropped the image to get relevant information about the street and perform zero mean normalization. I used this model to verify the implementation of generator functions, data augmentation and image cropping.
 The details of data augmentation and cropping are provided later.

Next I tried the Lenet model that I learnt in traffic signal classification project. The activation layer at the last stage was replaced with Dense(1) to fit the regression problem. The Lenet model worked for most part but the vehicle was going off the road at three places mentioned below. This happened even though validation loss was comparable to training loss.

Then I adopted even more power architecture suggested by autonomous vehicle group in Nvidia. The details of the model are in the picture below.  The Lambda layer was inserted to scale the data and realize zero mean normalization. The cropping layer was introduced to capture only the road information in the images.  

#### 2. Attempts to reduce overfitting in the model

The overfitting was a problem when large amount of data was captured in multiple laps.  So, I used only 2 and laps of data - first one with center driving, second one with vehicle recovering from the left and right and remaining frames to capture the bottle-necks in the course. I found three places where the vehicle had issues staying oncourse - so captured some more data at these bottlenecks.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and data from specific bottlenecks where the vehicle had issues staying on the tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use convolution neural network model Lenet that was successfully used in traffic signal classification problem. I thought this model might be appropriate because this model might come up with unique steering angle values depending on images/objects from the street. Soon I realized this model has difficulty in steering the vehicle at few places where the steering angle required were high or there was a dirt road that distracted the vehicle from staying on the right track/course.


Then I followed the lecture instructions and built the model based on Nvidia model. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set using 80/20 split. The first training set that I tried was one with center lane driving. I used 3 epochs for training and did not see any issue with overfitting (validation loss and training loss were comparable). The very first attempt was pretty good and I think it was bceause of data augmentation. 
    However, the vehicle fell off the track at 3-4 places (with different training attempts). 
 So, i added training data for another lap where i have several recoveries (both left and right) at the shoulders. Still the data was not sufficient as i did not see entire track/course completion with every training attempts.  Then I captured training data for 3 places where the vehicle had difficulty staying on the track.
  

To combat the overfitting, I avoided repeating training data from several laps with same type of driving. The data augmentation was done using image flipping, left camera image and right camera image. The choice of center, left, right and flip was done randomly and steering angle was modified according to the selection. This can be found in lines (24-42) of model.py.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and the issue was resolved by adding specific training data for those locations.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road. Here's a [link to my final video result](./run2.mp4)

#### 2. Final Model Architecture

The final model architecture (model.py lines 50-71) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on the track if it goes to the shoulders. These images show what a recovery looks like starting from left to right or right to left :

![alt text][image3]
![alt text][image4]


To augment the data sat, I also flipped images and angles and here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

The data is cropped using cropping2D() and cropped image is given below:
![alt text][image5]


Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by zero mean normalization and augmentation (use center, right, left and flip randomly)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as the validation loss starts oscillating beyond 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
