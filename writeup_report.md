# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image2]: ./examples/right2center_1.jpg "Recovery Image"
[image3]: ./examples/right2center_2.jpg "Recovery Image"
[image4]: ./examples/right2center_3.jpg "Recovery Image"
[image5]: ./examples/original.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network as below table (clone.py lines 219-248) 

| Layer         	 |     Description	        					| 
|:------------------:|:--------------------------------------------:| 
| Input        	     | 160x320x3 RBG image   						| 
| Cropping           | (70x25)(0,0) to 65x320x3                     |
| Lambda             | 255 --> -0.5 ~ 0.5                     |
| Convolution 5x5x24 | 5x5 stride(2,2), valid padding, outputs 30x158x24 |
| RELU				 |												|
| Convolution 5x5x36 | 5x5 stride(2,2), valid padding, outputs 13x77x36 |
| RELU				 |												|
| Convolution 5x5x48 | 5x5 stride(2,2), valid padding, outputs 5x37x48 |
| RELU				 |												|
| Convolution 5x5x64 | 5x5 stride(1,1), valid padding, outputs 1x33x64 |
| RELU				 |												|
| Fully connected	 | 1x33x64x3 -> 2112x3 -> 6336 					|
| Fully connected	 | 6336 -->120.        							|
| Fully connected	 | 120 -->60.        							|
| Fully connected	 | 60 -->12.        							|
| Fully connected    | 10 --> 1.        							|

The input data is (160x320x3) RBG images, first cropping y-axis on top 70 and bottom 25, then the output images changed to 65x320x3 (code line 233). And the data is normalized from 0~255 to -0.5 to 0.5 (code line 234).
The model includes 4 Convolution layers with RELU and first 3 layers stride(2,2) (code line 232 - 237), 
The model also includes 5 fully connected layers. (code line 238 to 242).

After that I used keras.compile() with loss='mse', optimizer = 'adam'. (code line 244)
for fit() funtion, seperated validation set as 20% of training sample. (code line 245)
Finally, save the training result as 'model.h5' (code line 248)

#### 2. Attempts to reduce overfitting in the model

I tried dropout() funtion in other 3 CNN models which are LeNet(code line 133 - 154) , LeNet-modified (code line 156 -186), created CNN (code line 188 - 217). But finally, both them were not good enough to run one lap of first track.

For the final CNN model, there is no reduce overfitting operation.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 246).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and muliple-time wrong place data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet, I thought this model might be appropriate because it is classicial CNN network. And I add dropout operation to avoid overfitting. But finally, the loss ratio was always larger than 0.04 and it was failed on autonomously driving. (code line 133 LeNet())

Secondly, I modified LeNet to reduce a fullconnection layer and change convolution layer size, and also add dropout processing. Then the loss ratio less than 0.03. In automomously mode, the vechile run far than LeNet. Unfortunitely it still failed to run a lap. (code line 156 LeNet_modified()).

I tried 3th CNN,it still not good enough. (code line 188 CNN_1())

I tried 4th CNN, it pass the automomously driving test. (code line 219). It is good enough even there is no dropout operation. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of a convolution neural network as below table (clone.py lines 219-248) 

| Layer         	 |     Description	        					| 
|:------------------:|:--------------------------------------------:| 
| Input        	     | 160x320x3 RBG image   						| 
| Cropping           | (70x25)(0,0) to 65x320x3                     |
| Lambda             | 255 --> -0.5 ~ 0.5                     |
| Convolution 5x5x24 | 5x5 stride(2,2), valid padding, outputs 30x158x24 |
| RELU				 |												|
| Convolution 5x5x36 | 5x5 stride(2,2), valid padding, outputs 13x77x36 |
| RELU				 |												|
| Convolution 5x5x48 | 5x5 stride(2,2), valid padding, outputs 5x37x48 |
| RELU				 |												|
| Convolution 5x5x64 | 5x5 stride(1,1), valid padding, outputs 1x33x64 |
| RELU				 |												|
| Fully connected	 | 1x33x64x3 -> 2112x3 -> 6336 					|
| Fully connected	 | 6336 -->120.        							|
| Fully connected	 | 120 -->60.        							|
| Fully connected	 | 60 -->12.        							|
| Fully connected    | 10 --> 1.        							|

The input data is (160x320x3) RBG images, first cropping y-axis on top 70 and bottom 25, then the output images changed to 65x320x3 (code line 233). And the data is normalized from 0~255 to -0.5 to 0.5 (code line 234).
The model includes 4 Convolution layers with RELU and first 3 layers stride(2,2) (code line 232 - 237), 
The model also includes 5 fully connected layers. (code line 238 to 242).

After that I used keras.compile() with loss='mse', optimizer = 'adam'. (code line 244)
for fit() funtion, seperated validation set as 20% of training sample. (code line 245)
Finally, save the training result as 'model.h5' (code line 248)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from wrong place. These images show what a recovery looks like starting from right to center:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would avoid always turn left. Due to the track one's center is in left. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After flip image, I also use left images and right images. And the measurement angle add 0.2 for left and reduce 0.2 for right. And I also found the left turn always not enough on curve road. Then I multiplied measurement angle by 1.2 to let left turn to normal in curve road.

I also found some place always had problem in autonomously driving mode. Then I record more driving data on the wrong place.

After the collection process, I had 20152 number of data points. I then preprocessed this data by Lambda(lambda x:(x/255.0)-0.5) to normalized the data to -0.5 to 0.5.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by the log and autonomously mode. I used an adam optimizer so that manually training the learning rate wasn't necessary.
