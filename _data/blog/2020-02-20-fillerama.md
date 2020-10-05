---
template: BlogPost
path: /Deep-learning
date: 2020-10-05
title: Malaria Cell Identification
thumbnail: /assets/output_9_1.png
---
# Maleria Detection

Hello! Here we are going to deal with identification of cell which are infected by Maleria. We have dataset in which we have images of cells
stored in two separate directories. One directory contains 13,780 images on infected cells and in another directory named as uninfected also contains
13,780 images of uninfected cells. 

So, in this notebook we are going to create a Deep Learning Model which will classify the cell image as infected on uninfected by marleria.

Here we will be dealing with images hence using a Convolutional Neural Network will be a first choice for any Machine Learning Practitioner. 
CNN will extract important features from the image, which will help our model to classify the particualar image as it is a 
infected or uninfected cell of Maleria.

## Collecting Data Set

This will be first step for every Machine Learning or Deep Learning Project. Without data, how will you train your model...

So let's download our dataset of Cell Images.


```python
!wget https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip
```

    --2020-10-04 14:41:51--  https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip
    Resolving ceb.nlm.nih.gov (ceb.nlm.nih.gov)... 130.14.52.15, 2607:f220:41e:7052::15
    Connecting to ceb.nlm.nih.gov (ceb.nlm.nih.gov)|130.14.52.15|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 353452851 (337M) [application/zip]
    Saving to: ‘cell_images.zip’
    
    cell_images.zip     100%[===================>] 337.08M  40.8MB/s    in 8.9s    
    
    2020-10-04 14:42:01 (37.8 MB/s) - ‘cell_images.zip’ saved [353452851/353452851]
    



```python
!unzip cell_images.zip
```

    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_118.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_126.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_134.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_141.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_168.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_175.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_183.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_221.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_222.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_87.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102428_cell_91.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102516_cell_104.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102516_cell_13.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102516_cell_146.png  
     extracting: cell_images/Uninfected/C236ThinF_IMG_20151127_102516_cell_168.png   
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_16.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_31.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_39.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_46.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_57.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_65.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_77.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_79.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141314_cell_81.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141351_cell_22.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141351_cell_26.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141351_cell_53.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141351_cell_56.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141351_cell_75.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141351_cell_83.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141351_cell_92.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_1.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_16.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_17.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_22.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_31.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_38.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_42.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_45.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_64.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_73.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_141520_cell_9.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_11.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_14.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_15.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_3.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_45.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_47.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_52.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_53.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_55.png  
     extracting: cell_images/Uninfected/C99P60ThinF_IMG_20150918_142128_cell_56.png  
      inflating: cell_images/Uninfected/Thumbs.db  


Above we downloaded the zip file that contains our data and we extracted it. As mentioned above we get two folders in ```cell_images``` folder, those are ```Parasitized``` and ```Uninfected```. ```Parasitized``` folder contains infected cell images having count of 13,780, similarly ```Uninfected``` folder contains uninfected mean healthy cell images having count of 13,780.


```python
Cell_Images_dir = '/content/cell_images'
Parasitized_dir = '/content/cell_images/Parasitized'
Uninfected_dir = '/content/cell_images/Uninfected'
```

Above we stored there the path of each folder so that it will be easier to use the path by directly calling the variables.

Now, Let's visualize what we have in our dataset. Let see one one images from 


```python
import os

Parasitized_images = os.listdir(Parasitized_dir)
Uninfected_images = os.listdir(Uninfected_dir)

```


```python
from PIL import Image

print('Parasitized Cell ')
Image.open(Parasitized_dir + '/C129P90ThinF_IMG_20151004_134306_cell_126.png')
```

    Parasitized Cell 





![png](/assets/output_9_1.png)




```python
print('Uninfected Cell ')
Image.open(Uninfected_dir + '/C142P103ThinF_IMG_20151005_223257_cell_174.png')
```

    Uninfected Cell 





![png](/assets/output_10_1.png)



After visualizing the Images from both classes we can easily detect that there is an infecter portion in Parasitized images. But we want to train our deep learning model that it should detect that and classify the images in Paracitized or Uninfected classes.

Now we have data which is in image format, but we have to convert it in machine tarinable format. We cannot input the image data as it is, we have to preprocess it first. The data we have is already supervised and separeted in folder so we have some functions from ```tensorflow``` which can easily preprocess these data.


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(validation_split=0.2 ,rescale=1/255.0)

cell_images_training =  data_gen.flow_from_directory(Cell_Images_dir, target_size=(125, 125), class_mode='categorical', subset='training')
cell_images_validation =  data_gen.flow_from_directory(Cell_Images_dir, target_size=(125, 125), class_mode='categorical', subset='validation')
```

    Found 22048 images belonging to 2 classes.
    Found 5510 images belonging to 2 classes.



```python
cell_images_training.class_indices
```




    {'Parasitized': 0, 'Uninfected': 1}



**If you are seeing this first time then I am sure it's little confusing, Let me explain what it is and what is it doing?**

First we imported a ```ImageDataGenerator``` class from ```keras``` which helps us to preprocess images so that we can use them to train our neural network.  

Next we created a object of ```ImageDataGenerator``` class and passed a ```Normalization Factor``` i.e. ```1/255.0```. 
Normalization is a process that changes the range of pixel intensity values. If you want to read more about normalization then [Click Here](https://en.wikipedia.org/wiki/Normalization_(image_processing)).

There is one more parameter that we passed, which is ```validation_split```. It will split our data in two datasets ```training``` and ```testing```. 

After creating the object, now we are able to access the functions of it. There is function ```flow_from_directory``` which goes through the directory and catogorixe the data according to the folders in passed directory. Let's understand about the parameters that are passed in it:
* ```Cell_Image_dir``` - This is a variable where path of main directory is stored. This directory contains training images of cells classified in two different folders as ```Parasitized``` and ```uninfected```.
* ```target_size=(125, 125)``` - It converts all images in same size as passed. Here I passes (125, 125). In Neural Networks it is very important that all input images must have same size. 
* ```class_mode='categorical'``` -  It specifies the type of classification that must be done on passed directory. 
* ```subset='training'``` / ```subset='validation'``` - It will classify datasets into training and validation dataset.

If you want to read more about Image Processing in Keras then [Click Here](https://keras.io/api/preprocessing/image/)

Now we are ready with our Data. Preprocessing of our data is handled by Keras. Now we can work on creating Nerural Network and traing our data on that model. 

Here we will be using Keras for creating our Deep Neural Network. We will be dealing with image data hence we will use Convolutional Neural Network to train our data. 


Let's import the necessary packages, then we will discuss about each of them. 


```python
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
```

Above we imported different layers and ```Sequential``` model from Keras. Each layer has different tasks to perform.  

Layers that we imported:
* ```Conv2D``` - This layer creates a convolution kernel that is convolved with the layer input to extract features from inputed image.
* ```MaxPool2D``` - Downsamples the input representation by taking the maximum value over the window defined by ```pool_size``` for each dimension along the features axis.
* ```Dense``` - Dense layer adds another layer in Neural Network with specific activation function.
* ```Dropout``` - The Dropout layer randomly sets input units to 0, which helps in overfitting the model
* ```Flatten``` - Flattens the input. Converts input size to 1 dimension. 

* ```Sequential``` - A Sequential model is appropriate for a plain stack of layers. Here we will be dealing continual series of Layers. 

Now, let's create our model for classification of infected Malaria cell images.  


```python
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(125, 125, 3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

Above we have created our Neural Network which will be used to classify the infected Malaria Cell Images. 

First we created ```model``` as a object of ```Sequential``` class. Here our neaural network gets initiated. 

Then we will ```add``` layers to the neural network, there positions are pre-decided as per their functionality.

Whole Neural Network is divided into two sections: 
1. Feature Learning
2. Classification


1. Feature Learning- 

  Here we will be extracting the important features from the image to make perfect classification. In this layer we have ```Conv2D``` and ```MaxPool2D``` layers. 
  * ```Conv2D``` - As we know this is a convolutional layer. It creates kernels and convolve it throughout the image which focuses on the particular feature of image as per  the values in the kernel. Keras takes care of defining kernel with different values. Let's discuss about the parameters that we passed in convolutional layer.
    * ```64``` - The first parameter in Conv2D layer is the number of kernels that we want to be convolved throughout the image. 
    * ```(3, 3)``` - The second parameter is the size of that kernel.
    * ```activation='relu'``` - Here we pass the activation function to convert the result of convolutin in binary. 
    * ```input_shape = (125, 125, 3)``` - Here we describe the shape of the input image. This we need to specify in only first layer.
  
  * ```MaxPool2D``` - This is a Pooling Layer, this is used to decrease some shape of the image and extract only maximum value in particular window. 
    * ```pool_size = (2, 2)``` - Here we pass the size of pool window. This window in revolved throughout the convolved image and extracts only the maximum value.
  
  These two layers are used to extract important features for perfect classifiaction.
  
2. Classification - 

  In this section Neural Network focuses on the classification part. Here we have ```Flatten``` and ```Dense``` Layers. 
  * ```Flatten``` - This layer is used for dimentionality reduction. This reduces the dimention of the input data and convert it into searies of features. 
  * ```Dense``` - Dense layer just adds another neural layer in the network, which is also called fully connected layer, which important to connect the flatten layer with output layer.

In Neural Network we can also see one layer named ```Dropout```. What this layer does is that it randomly sets input to zero. It is used to avoid overfitting of the model. 



This is the overview of our neural network, but it only tells that what it is used for but if you want to see what is happening inside, then we have a function ```summary()``` which gives detailed overview of our model.  


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 123, 123, 64)      1792      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 61, 61, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 59, 59, 64)        36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 29, 29, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 27, 27, 128)       73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 128)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 11, 11, 128)       147584    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 5, 5, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 3200)              0         
    _________________________________________________________________
    dropout (Dropout)            (None, 3200)              0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               1638912   
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 1026      
    =================================================================
    Total params: 1,900,098
    Trainable params: 1,900,098
    Non-trainable params: 0
    _________________________________________________________________


As you can see above, it gives the complete overview of the our Neural Network. It tell that in our network we arew dealing with nearly 1 million neurons.

So now let train the model with the Dataset that we generated.


```python
H = model.fit(cell_images_training, epochs=10, validation_data=cell_images_validation, verbose=1)
```

    Epoch 1/10
    689/689 [==============================] - 36s 52ms/step - loss: 0.3346 - accuracy: 0.8460 - val_loss: 0.2131 - val_accuracy: 0.9376
    Epoch 2/10
    689/689 [==============================] - 33s 48ms/step - loss: 0.1873 - accuracy: 0.9445 - val_loss: 0.2404 - val_accuracy: 0.9397
    Epoch 3/10
    689/689 [==============================] - 34s 49ms/step - loss: 0.1802 - accuracy: 0.9473 - val_loss: 0.2172 - val_accuracy: 0.9416
    Epoch 4/10
    689/689 [==============================] - 34s 49ms/step - loss: 0.1744 - accuracy: 0.9495 - val_loss: 0.1930 - val_accuracy: 0.9428
    Epoch 5/10
    689/689 [==============================] - 33s 48ms/step - loss: 0.1763 - accuracy: 0.9487 - val_loss: 0.2269 - val_accuracy: 0.9397
    Epoch 6/10
    689/689 [==============================] - 33s 48ms/step - loss: 0.1716 - accuracy: 0.9506 - val_loss: 0.2904 - val_accuracy: 0.9448
    Epoch 7/10
    689/689 [==============================] - 33s 49ms/step - loss: 0.1743 - accuracy: 0.9501 - val_loss: 0.2117 - val_accuracy: 0.9381
    Epoch 8/10
    689/689 [==============================] - 33s 48ms/step - loss: 0.1701 - accuracy: 0.9494 - val_loss: 0.2893 - val_accuracy: 0.9316
    Epoch 9/10
    689/689 [==============================] - 33s 48ms/step - loss: 0.1747 - accuracy: 0.9500 - val_loss: 0.1915 - val_accuracy: 0.9379
    Epoch 10/10
    689/689 [==============================] - 33s 48ms/step - loss: 0.1761 - accuracy: 0.9492 - val_loss: 0.1993 - val_accuracy: 0.9412


Above we trained the model with our generated Dataset. Let's discuss about the parameters that we passed to the ```fit``` :
* ```cell_images_training```-  This is the training data that we have generated preveously from our Data Directory.
* ```epochs=10``` - Epochs is a number that how many times model is going to train on that training data. For each epoch Network Backtrack the model and improves the accuracy. 
* ```validation_data=cell_images_validation``` - Here we will pass our testing data.
* ```verbose=1``` - Due to this, it displays the output.

Object of this whole process is stored in the variable ```H```.
From this object we can get values of ```training accuracy```, ```validation accuracy```, ```training loss``` and ```validation loss``` for each epochs. 

And by accessing these information we will plot a training and validation graph for loss and accuracy of our model.


```python
import numpy as np
import matplotlib.pyplot as plt


acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
loss = H.history['loss']
val_loss = H.history['val_loss']
n =np.arange(1,11)  # Because 10 epochs
```


```python
fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(n, loss, label = 'Training Loss')
ax1.plot(n, val_loss, label = 'Val_Loss')
ax1.legend()

ax2.plot(n, acc, label = 'Training Accuracy')
ax2.plot(n, val_acc, label = 'Val Accuracy')
ax2.legend()

plt.show()
```


![png](/assets/output_26_0.png)


The graph explains that Validation Accuracy is increasing gradually and at on point it becomes constant. So it tells that our models is quite good. 

So now let save the model, so that we can easily share it with others or deploy to make it available to others.


```python
model.save('maleria_cell_classyfier.h5')
```
