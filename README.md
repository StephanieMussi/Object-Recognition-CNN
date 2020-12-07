# Object_Recognition_CNN
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset contains 32x32 RGB color images each labeled from 0 to 9. In this project, batch 1 of the train data is used, the the test data is trimmed to 2000 samples. The trimmed test data can be found in ("test_batch_trim.csv")[https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/test_batch_trim].  
## Initial Model
The Architecture of the Convolution Network Model implemented is as below:  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/CNNmodel.png" width = 800 height = 400>  

The codes for the model construction is as followed: 
```python
 model = tf.keras.Sequential()
    model.add(layers.Input(shape=(3072, )))
    model.add(layers.Reshape(target_shape=(32, 32, 3), input_shape=(3072,)))
    model.add(layers.Conv2D(num_ch_c1, kernel_size=9, padding='valid', activation='relu', use_bias=True, input_shape=(None, None, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(layers.Conv2D(num_ch_c2, kernel_size=5, padding='valid', activation='relu', use_bias=True))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, use_bias=True, input_shape=(300,)))
```
After training with SGD optimizer on mini-batches of size 128 for 1000 epochs, the accuracies are:  
|Train Accuracy | Test Accuracy |
|:-:|:-:|
|74.68% |49.55%| 
  
The graphs of accuracies and losses are plotted:  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/SGDAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/SGDLoss.png" width = 300 height = 200>   

In order to see if the model is actually extracting information from the images, the feature maps of the first input image is displayed:  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/1st_input.png" width = 320 height = 320>  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/conv1_fm.png" width = 500 height = 300>
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/maxp1_fm.png" width = 500 height = 300>  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/conv2_fm.png" width = 500 height = 300>
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/maxp2_fm.png" width = 500 height = 300>  
  
As it can be seen, the feature maps of the latter layers are more blurred and abstract, which means the model is learning higher level features to classify the image.  

  
## Find Optimal Number of Channels  
There are C1 channels in the first convolutional layer and C2 channels in the second. For C1 = 10, 30, 50, 70, 90 and C2  = 20, 40, 60, 80, 100, the model is trained and the test accuracies are as below:  

| |C2 = 20	|C2 = 40|	C2 = 60	|C2 = 80|	C2 = 100|
|:-:|:-:|:-:|:-:|:-:|:-:|
|C1 = 10|	0.4645|	0.4320|	0.4795|	0.4665|	0.4865|
|C1 = 30|	0.4625|	0.4650	|0.4695|	0.4905|	0.4940|
|C1 = 50	|0.4775|	0.4665	|0.4955|	0.5025|	0.4870|
|C1 = 70|	0.4885|	0.4940|	0.4950|	0.5000|	0.5110|
|C1 = 90	|0.4845	|0.5095|	0.5055	|0.5125|	0.5200|  
  
As can be seen, the combinition of C1 = 90 and C2 = 100 yields the highest accuracy. In this case, the train and test accuracies are:  
|Train Accuracy | Test Accuracy |
|:-:|:-:|
|81.97% |52.00%| 
  

## Find Best Optimizer
In the model above, the basic SGD optimizer is used. In order to boost performance, some adjustment is made, and the performance is compared.  
The following codes are used to switch between different optimizers:  
```python
    optimizer_ = 'SGD' #'Adam' #'RMSProp' #'SGD-momentum' '    
    if optimizer_ == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_ == 'SGD-momentum': 
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum = 0.1)
    elif optimizer_ == 'RMSProp':  
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_ == 'Adam': 
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise NotImplementedError(f'You do not need to handle [{optimizer_}] in this project.')
```
### Use SGD Momentum
Using momentum of 0.1, the accuracies obtained are:  
|Train Accuracy | Test Accuracy |
|:-:|:-:|
|87.01% |52.05%|  
  
The accuracy and loss graphs are:  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/SGDMAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/SGDMLoss.png" width = 300 height = 200>   

It can be seen that adding momentum speeds up the learning process.   

### Use RMSProp
Using RMSProp algorithm, the accuracies obtained are:  
|Train Accuracy | Test Accuracy |
|:-:|:-:|
|99.74% |45.55%|  
  
The accuracy and loss graphs are:  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/RMSAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/RMSLoss.png" width = 300 height = 200>   

It is shown that the accuracies converges fast and too many epochs lead to serious overfitting. Early-Stopping should be applied, which means the training should stop when the test loss is the smallest.   

### Use Adam
Using Adam optimizer, the accuracies obtained are:  
|Train Accuracy | Test Accuracy |
|:-:|:-:|
|100% |40.20%|  
  
The accuracy and loss graphs are:  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/AdamAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/AdamLoss.png" width = 300 height = 200>  

Similar to RMSProp, the accuracies with Adam converge extremely fast, and periodically drop to a very low value.  


## Use dropout
Using the basic SGD optimizer, dropout layers with rate of 0.5 are added to the model.  
```python
if use_dropout:
        model.add(layers.Dropout(rate = 0.5))
```  
The accuracies obtained are:  
|Train Accuracy | Test Accuracy |
|:-:|:-:|
|59.13% |51.35%|  
  
The accuracy and loss graphs are:  
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/DOAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/DOLoss.png" width = 300 height = 200>  
Adding dropout layer leads to a lower convergent accuracy but reduces overfitting at the same time. As can be seen, the difference between train result and test result are not as big. The reason is that only a selective set of weights are used for learning.  
