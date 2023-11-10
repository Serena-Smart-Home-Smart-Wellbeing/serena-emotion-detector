#!/usr/bin/env python
# coding: utf-8

# # Datasets

# Run this `gcsfuse` cell if you can't list the folders inside of "/gcs"

# In[2]:




# When using GCS buckets, use "/gcs" instead of "gs://"

# In[3]:


#from sklearn.model_selection import train_test_split

dataset_path = "/gcs/serena-shsw-datasets"
training_dataset = dataset_path + "/FER-2013/train"
test_dataset = dataset_path + "/FER-2013/test"

# Split the dataset into training and validation sets
#training_dataset, validation_dataset = train_test_split(training_dataset, test_size=0.2, random_state=42)

# Output directory contents


# # Import Library

# In[4]:


import csv
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# # Create Model

# In[5]:


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[6]:


num_classes = 7  # Gantikan dengan jumlah sebenarnya dari kelas emosi dalam dataset Anda
batch_size = 32 * 4 # 128 seems to be ideal
num_epochs = 10
train_data_dir = 'path_to_train_data_directory'
validation_data_dir = 'path_to_validation_data_directory'
test_data_dir = 'path_to_test_data_directory'


# In[7]:


#num_classes = 7  # Gantikan dengan jumlah sebenarnya dari kelas emosi dalam dataset Anda

#x = layers.GlobalAveragePooling2D()(x)
#x = layers.Dense(1024, activation='relu')(x)
#predictions = layers.Dense(num_classes, activation='softmax')(x)

#model = models.Model(inputs=base_model.input, outputs=predictions)


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


# # Data Augmentation

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
    batch_size=64,
    class_mode='categorical'
)


# # Data Generating

# In[10]:


validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

validation_generator = validation_datagen.flow_from_directory(
    directory=test_dataset,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical'
)


# In[11]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# history = model.fit(
#     train_generator,
#     epochs=10, 
#     validation_data=validation_generator,
# )

# In[12]:


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
)


# # Saving Model

# Vertex AI expects the model artifacts to be saved in `BASE_OUTPUT_DIRECTORY/model/` when you want to train a new version of a model

# In[13]:


saved_model_path = dataset_path + "/models/serena-emotion-detector/model"


# In[14]:


model.save(saved_model_path)


# # Evaluate Model

# In[15]:


saved_emotion_detector = tf.keras.models.load_model(saved_model_path)

# Check its architecture
saved_emotion_detector.summary()


# In[17]:


val_loss, val_accuracy = model.evaluate(training_datasets, training_labels)
print(f'Validation Accuracy: {val_accuracy}')

test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {test_accuracy}')

