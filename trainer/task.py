#!/usr/bin/env python
# coding: utf-8

# # Datasets

# Run this `gcsfuse` cell if you can't list the folders inside of "/gcs"

# In[1]:




# When using GCS buckets, use "/gcs" instead of "gs://"

# In[2]:


#from sklearn.model_selection import train_test_split

dataset_path = "/gcs/serena-shsw-datasets"
training_dataset = dataset_path + "/FER-2013/train"
test_dataset = dataset_path + "/FER-2013/test"
validation_dataset = dataset_path + "/FER-2013/valid"

# Split the dataset into training and validation sets
#training_dataset, validation_dataset = train_test_split(training_dataset, test_size=0.2, random_state=42)

# Output directory contents


# # Import Library

# In[5]:


import csv
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# # Hyperparameters

# In[6]:


num_classes = 7  # Gantikan dengan jumlah sebenarnya dari kelas emosi dalam dataset Anda
batch_size = 32 * 4 # 128 seems to be ideal
epochs = 10


# # Create Model

# In[7]:


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[6]:


#num_classes = 7  # Gantikan dengan jumlah sebenarnya dari kelas emosi dalam dataset Anda

#x = layers.GlobalAveragePooling2D()(x)
#x = layers.Dense(1024, activation='relu')(x)
#predictions = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)


# In[8]:


for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])


# # Data Generating

# In[9]:


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=training_dataset,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

validation_generator = validation_datagen.flow_from_directory(
    directory=test_dataset,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)


# In[10]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[11]:


history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
)


# # Saving Model

# Vertex AI expects the model artifacts to be saved in `BASE_OUTPUT_DIRECTORY/model/` when you want to train a new version of a model

# In[12]:


saved_model_path = dataset_path + "/models/serena-emotion-detector/model"

# Do not uncomment this line, it will be done by setup.sh
model.save(saved_model_path)

