#**Traffic Sign Recognition** 

[//]: # (Image References)
[image0]: ./examples/intro.png "Introduction"
[image1]: ./examples/lenet.png "LeNet"
[image2]: ./examples/bar_chart_train_dataset.png "Train Distribution Set"
[rslt_stop]: ./examples/result_stop.png "Result Probability Stop Sign"
[rslt_priority]: ./examples/result_priority_road.png "Result Probability Priority Road"
[rslt_50kph]: ./examples/results_50kph.png "Result Probability Speed Limit 50kph"
[rslt_roadwork]: ./examples/results_road_Work.png "Result Probability Road Work"
[rslt_General_Caut]: ./examples/result_generalcaution.png "Result Probability General Caution"
[Grayscale]: ./examples/grayscale.png "Grayscale"
[HSV]: ./examples/HSV.png "HSV"
[YUV]: ./examples/YUV.png "YUV"
[Rotate]: ./examples/Rotate.png "Rotate"
[Enhance]: ./examples/Enhance.png "Enhance"
[Translate]: ./examples/Translate.png "Translate"


-> ![Introduction][image0] <-

This project builds a Traffic sign recognition classifier to identify German traffic sign. In this project a [LeNet-5](http://yann.lecun.com/exdb/lenet/)  architecture propsed by Yann LeCun. This architecture is a conventional neural netwrok that was designed to recogonise the hand written visual patterns from the image with minimal preprocessing.

-> ![LeNet][image] <-

Source: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
---

The following steps are used to create the classfier, pipeline and training process.

- Load the data.
- Understanding and Visualising the data.
- Define training set, validation set and test set.
- Design of Pipeline.
- Training of Network.
- Run Benchmark model without any preprocessing.
- Preprocessing the data for usage.
    - Use of different normalisation methods.
    - Apply different image augmentation methods.
- Measure of system performance.
- Tunning of system performance.
- Run classifier on Test data.


###1. Data Set Summary & Exploration

The given data set is database of German traffic signs collected for a project run by the institute for Informatics [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The data provide in the course are pickled python object which contains a dictionary of images and its labels. The data is loaded using the pickle module and allocated as original data set. The data is first analysed to understand the image size, type and dataset information.


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Training dataset Distribution of traffic signs][[image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

[Grayscale]: ./examples/grayscale.png "Grayscale"
[HSV]: ./examples/HSV.png "HSV"
[YUV]: ./examples/YUV.png "YUV"
[Rotate]: ./examples/Rotate.png "Rotate"
[Enhance]: ./examples/Enhance.png "Enhance"
[Translate]: ./examples/Translate.png "Translate"

![Gray Scale][Grayscale]


![HSV Colour Space][HSV]


![YUV Colour Space][YUV]


![Rotate Colour Space][Rotate]


![Enhance Colour Space][Enhance]


![Translate Colour Space][Translate]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer           	| Shape    	| Description               	|
|:----------------:	|:---------:|:--------------------------:	|
| Input           	| 32x32x3  	| RGB Image                 	|
| Convolution     	| 28x28x6  	| 1x1 Stride, Valid Padding 	|
| Activation      	| 28x28x6  	| ReLU                      	|
| Max Pooling     	| 14x14x6  	| 2x2 Stride, Valid Padding 	|
| Convolution     	| 10x10x16 	| 1x1 Stride, Valid Padding 	|
| Activation      	| 10x10x16 	| ReLU                      	|
| Max Pooling     	| 5x5x16   	| 2x2 Stride, Valid Padding 	|
| Flatten         	| 400      	|                           	|
| Fully Connected 	| 120      	|                           	|
| Activation      	| 120      	| ReLU                      	|
| Fully Connected 	| 84       	|                           	|
| Activation      	| 84       	| ReLU                      	|
| Fully Connected 	| 43       	|                           	|
| Softmax         	| 43       	| ReLU                      	|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 50 Kmph 	| Speed Limit 50 Kmph							| 
| General Caution		| General Caution								|
| Priority Road			| Priority Road                 				|
| Road Work	      		| Pedestrians					 				|
| Stop Sign 			| Stop Sign         							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 119, 121 and 123rd resultcell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Prediction            | Probability   |
|:---------------------:|:-------------:|
| Speed limit (50km/h)  | 0.99996674061 |
| Speed limit (80km/h)  | 0.00003324157 |
| Speed limit (100km/h) | 0.00000000000 |
| Speed limit (30km/h)  | 0.00000000000 |
| Speed limit (60km/h)  | 0.00000000000 |

![Softmax Probability of Speed Limit 50kph][rslt_50kph]

Second Image

| Prediction           | Probability   |
|:--------------------:|:-------------:|
| Traffic signals      | 0.99986636639 |
| General caution      | 0.00013365022 |
| Pedestrians          | 0.00000000000 |
| Speed limit (20km/h) | 0.00000000000 |
| Speed limit (30km/h) | 0.00000000000 |


![Softmax Probability of General Caution][rslt_General_Caut]

Third Image

| Prediction           | Probability  |
|:--------------------:|:------------:|
| Priority road        | 1.0000000000 |
| Speed limit (20km/h) | 0.0000000000 |
| Speed limit (30km/h) | 0.0000000000 |
| Speed limit (50km/h) | 0.0000000000 |
| Speed limit (60km/h) | 0.0000000000 |

![Softmax Probability of Priority Road][rslt_priority]

Fourth Image

| Prediction                            | Probability |
|:-------------------------------------:|:-----------:|
| Road work                             | 1.000000000 |
| Right-of-way at the next intersection | 0.000000000 |
| Turn right ahead                      | 0.000000000 |
| Children crossing                     | 0.000000000 |
| Road narrows on the right             | 0.000000000 |

![Softmax Probability of Roadwork][rslt_roadwork]

Fifth Image

| Prediction                                   | Probability   |
|:--------------------------------------------:|:-------------:|
| Stop                                         | 0.99997258186 |
| No entry                                     | 0.00002637984 |
| Speed limit (120km/h)                        | 0.00000104451 |
| No passing                                   | 0.00000000176 |
| No passing for vehicles over 3.5 metric tons | 0.00000000002 |

![Softmax Probability of Stop][rslt_stop]




