#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets

from PIL import Image


# https://www.kaggle.com/datasets/trolukovich/food11-image-dataset

# In[27]:


# class 종류
os.listdir('food/training')


# In[28]:


train_paths = glob('food/training/*/*')
validation_paths = glob('food/validation/*/*')
test_paths = glob('food/evaluation/*/*')


# In[29]:


path=train_paths[0]
path


# In[30]:


gfile=tf.io.read_file(path)
image=tf.io.decode_image(gfile)
image.shape


# In[31]:


plt.imshow(image)
plt.show()


# ## Data Generator

# In[32]:


# 경로
train_dir = 'food/training'
validation_dir = 'food/validation'
test_dir = 'food/evaluation'


# In[76]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[77]:


target_size=(256,256)
batch_size=32


# In[78]:


# image 변환
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True # 수평 뒤집기
)

# 이미지를 batch size만큼 불러와줌.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True
)


# In[79]:


validation_datagen = ImageDataGenerator(
    rescale=1./255.
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)


# In[80]:


test_datagen = ImageDataGenerator(
    rescale=1./255.
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)


# ## Model

# In[81]:


input_size=(256,256,3)
dropout_rate=0.7 
num_classes=len(os.listdir('food/training'))
learning_rate=0.001 # 학습률


# In[82]:


inputs = layers.Input(input_size)

# Feature Extraction
net = layers.Conv2D(16,(3,3), padding='SAME')(inputs)
net = layers.MaxPooling2D(pool_size=(2,2))(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32,(3,3), padding='SAME')(net)
net = layers.MaxPooling2D(pool_size=(2,2))(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(dropout_rate)(net)  # 과적합 방지 (학습시)

# Fully Connected
net = layers.Flatten()(net)
net = layers.Dense(100)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(dropout_rate)(net)
net = layers.Dense(num_classes)(net)  # 마지막 layer는 class의 개수만큼
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net)


# In[85]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# ## Training

# In[87]:


history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)


# In[88]:


history.history.keys()


# In[91]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.legend(['train','validation'])
plt.show()


# In[92]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show()


# ## evaluation

# In[116]:


scores = model.evaluate_generator(
    test_generator,
    steps=len(validation_generator)
)


# In[121]:


print("accuracy :",scores[1]*100,"%")


# ## Predict

# In[99]:


path=test_paths[0]
path


# In[100]:


gfile = tf.io.read_file(path)
image = tf.io.decode_image(gfile)
image.shape


# In[101]:


image=image[tf.newaxis, ...]
image.shape


# In[104]:


from PIL import Image


# In[135]:


output = model.predict_generator(test_generator,steps=len(test_generator))


# In[123]:


output

