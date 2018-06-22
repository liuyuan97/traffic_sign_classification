#**Traffic Sign Recognition** 


[//]: # (Image References)
[image1]: ./training.png "training hist"
[image2]: ./validation.png "validation hist"
[image3]: ./test.png "test hist"

[image4]: ./traffic-signs-data/sign_limit80_5.jpg "Traffic Sign 1"
[image5]: ./traffic-signs-data/sign_no_entry_17.jpg "Traffic Sign 2"
[image6]: ./traffic-signs-data/sign0_roadWork_25.jpg "Traffic Sign 3"
[image7]: ./traffic-signs-data/sign3_stop_14.jpg "Traffic Sign 4"
[image8]: ./traffic-signs-data/sign-children-traffic-28.jpg "Traffic Sign 5"


###Data Set Summary & Exploration

The traffic signs data set is in the binary format of pickle.  To be more specific, 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 by 32
* The number of unique classes/labels in the data set is 43

The histograms of the training, validation, and test data are shown as  
![alt text][image1]  
![alt text][image2]   
![alt text][image3]   

However, the sign data distribution is not uniform.  Some sign classes are of small amount.  This might lead to the poor classification results for these signs.

###Design and Test a Model Architecture


####1. Data Preprocess

A simple data proproses approach is chosen, in which each channel of the RGB data is normalized in the range of -1.0-1.0.  Since the RGB data channel data is unsigned 8-bit integer value whose range is from 0 to 255.  Thus, this data preprocess is done by substrate these values by 128.0 and dividing these results by 128.0.

The data is not converted to the gray scale value as people normally do.  The underlying reason is to conserve the color information which might lead to a better performance if a proper network architecture is developed. 


####2. The Proposed CNN architecture

 Layer                 |     Description	        					
 ---------------------|---------------------------------------------  
 Convolution Layer with ELU and Max pooling | input 32x32x3, outputs 28x28x9, no padding, kernel size 5x5  stride 1x1   
 Convolution Layer with ELU and Max pooling | input 14x14x9, outputs 10x10x24, no padding, kernel size 5x5  stride 1x1   	
 Fully Connected Layer with RELU and drop	| input 600, outputs 400, drop rate 0.5							
 Fully Connected Layer with RELU 	        | input 400, outputs 120		
 Fully Connected Layer with RELU 	        | input 120, outputs 43				


####3. Training Parameters

* The batch size is 256
* The number of epochs is 10
* Learning rate is 0.002
* Adam optimizer is used

####4. Training Process

* The model starts with digit classification CNN mode with the change to handle RGB image instead of grayscle image
* Reduce the learning rate since the old one seems too aggressive to achieve better final accuracy
* Since the classified objects are more, increasing batch size to be 256 for fast convergence
* Add the drop layer to combat overfitting
* Use ELU instead of RELU for fast convergence
* Increase CNN parameter to accommodate more classification objects, and RGB image  


####5. Final Results 

The final model results are:
* validation set accuracy of 95.9% 
* test set accuracy of 94.0%
 

###Test a Model on New Images

####1. Five German traffic signs found on the web 

**Speed Limit (80km/h)**  
![alt text][image4]   
   
**No Entry**  
![alt text][image5]   

**Road Work**   
![alt text][image6]   

**Stop**   
![alt text][image7]   
   
**Chrildren Crossing**   
![alt text][image8]

The first image might be difficult to classify because there is a projection angle to take that pic.  This project might cause the picture pixel distortion.  If there is not such data in the training data, the propose CNN might not be able classify it. 

####2. Classification Results.

Here are the results of the prediction:

| Image | Prediction        					  
| ----- | -----------  
80 km/h	| Yield 
Road work | Road work 	
No entry  | No entry
Stop sign | Stop sign		 
Children traffic | Children traffic     		 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 91.6%

####3.Top 5 candidates for each image 

The top 5 candidates for each image classification are also analyzed. The detail analysis is as following.

For the first image, the model is relatively sure that this is a yield sign (probability quite achieving 1).  However, the correct classification is included in the top 5 candidate even though the probability is quite low.

For the second, third, and fifth images, the model is relatively sure that these are correct with very high scale close to one.

Only for the forth image, the model is not very sure about the classification result.

### Visualizing the Neural Network 

The second downloaded picture, No Entry, is used as the input image to view the convolution layer output features.

####1. Convolution Lay 1 output
All these 8 feature maps seem to capture the bar feature and the round shape feature in the picture.

####2. Convolution Lay 2 output
These feature maps do not clearly show any obvious feature which is directly related with the original picture.  This behavior matches with the normal CNN characteristics, i.e. the beginning layers tends to extract features.