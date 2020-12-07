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
<img src = "https://github.com/StephanieMussi/Object_Recognition_CNN/blob/main/Figures/max2_fm.png" width = 500 height = 300>  
  

## Find Optimal Number of Channels
## Find Best Optimizer
### Use SGD Momentum
### Use RMSProp
### Use Adam
## Use dropout
