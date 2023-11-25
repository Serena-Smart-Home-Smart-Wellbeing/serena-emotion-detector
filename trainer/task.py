#!/usr/bin/env python
# coding: utf-8

# This notebook is only for developing, training, and saving the model. For evaluating the model, check `evaluate.ipynb`.

# # Datasets

# Run this `gcsfuse` cell if you can't list the folders inside of "/gcs"

# In[9]:




# When using GCS buckets, use "/gcs" instead of "gs://"

# In[10]:


#from sklearn.model_selection import train_test_split

dataset_path = "/gcs/serena-shsw-datasets"
training_dataset = dataset_path + "/FER-SERENA/train/train"
test_dataset = dataset_path + "/FER-SERENA/test/test"
validation_dataset = dataset_path + "/FER-SERENA/valid/validation"

# Output directory contents


# # Import Library

# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input  # Import Input layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf


# # PRE-PROCESSING DATA

# In[12]:


# Define ImageDataGenerator for data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[13]:


img_size = (48, 48)
batch_size = 64

train_generator = train_datagen.flow_from_directory(
    training_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)


# # Create Model

# In[14]:


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

model.summary()


# # Data Generating

# In[15]:


# Compile your model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=200,
)


# # Saving Model

# Vertex AI expects the model artifacts to be saved in `BASE_OUTPUT_DIRECTORY/model/` when you want to train a new version of a model

# In[1]:


saved_model_path = dataset_path + "/models/serena-emotion-detector/model"

# Do not uncomment this line, it will be done by setup.sh
model.save(saved_model_path)


# After saving, use `evaluate.ipynb` to evaluate the model after loading the artifacts.
