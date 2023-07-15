#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.getcwd()
os.chdir(r'C:\Users\windows\Downloads')


# In[3]:


import tensorflow as  tf
from keras.preprocessing.image import ImageDataGenerator


# In[18]:


## Preprocessing the Training Set 
train_datagen = ImageDataGenerator(
    rescale =1./255, ## Feature scaling to get individual pixels 
    shear_range = 0.2, # Geometric transformations to reduce overfitting 
    zoom_range = 0.2,  # "" ^
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'chest_xray/train', 
    target_size = (64,64),
    batch_size = 32, 
    class_mode = 'binary')


# In[19]:


## Preprocessing the test set 

test_datagen = ImageDataGenerator(1./255) 
test_set = test_datagen.flow_from_directory(
    'chest_xray/test', 
    target_size = (64,64),
    batch_size = 32, 
    class_mode = 'binary')


# In[34]:


## Building the CNN 

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3, activation='relu', input_shape=[64,64,3])) ## Change the filter if needed to investigate different results

## Pooling 

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

## 2nd Layer 
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3, activation='relu')) ## Change the filter if needed to investigate different results
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

## Flattening 

cnn.add(tf.keras.layers.Flatten())

## Full Connection 

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu')) ## More units might yield better results 

## Output Layer 

cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) ## Binary classification = sigmoid activation, 1 neuron in output layer



# In[35]:


## Training the CNN 
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[ 'accuracy' ])


# In[36]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 27)


# In[23]:


## Making a single Prediction 

import numpy as np 
from keras.preprocessing import image 

test_image = tf.keras.preprocessing.image.load_img('chest_xray/single_prediction/person1_bacteria_1.jpeg', target_size = (64,64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices ## Tells us which index corresponds to what 'value'
if result[0][0] == 0:
    prediction = 'This patient is healthy'
else: 
    prediction = 'This patient has pneumonia'


# In[24]:


print(prediction)


# 

# In[28]:


m = tf.keras.metrics.FalseNegatives()
m.result().numpy()

cnn.summary()


# In[ ]:




