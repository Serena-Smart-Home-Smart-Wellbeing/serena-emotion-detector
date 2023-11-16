#!/usr/bin/env python
# coding: utf-8

# This notebook is only for developing, training, and saving the model. For evaluating the model, check `evaluate.ipynb`.

# # Datasets

# Run this `gcsfuse` cell if you can't list the folders inside of "/gcs"

# In[2]:




# When using GCS buckets, use "/gcs" instead of "gs://"

# In[3]:


#from sklearn.model_selection import train_test_split

dataset_path = "/gcs/serena-shsw-datasets"
training_dataset = dataset_path + "/FER-2013/train"
test_dataset = dataset_path + "/FER-2013/test"
validation_dataset = dataset_path + "/FER-2013/valid"

# Split the dataset into training and validation sets
#training_dataset, validation_dataset = train_test_split(training_dataset, test_size=0.2, random_state=42)

# Output directory contents


# # Import Library

# In[4]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from google.cloud import aiplatform


# # PRE-PROCESSING DATA

# In[5]:


BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 7  # Anggap ada 7 kelas emosi wajah


# In[6]:


train_data_path = training_dataset
test_data_path = test_dataset


# # Create Model

# In[7]:


# Gunakan ImageDataGenerator untuk augmentasi data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    train_data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_dataset = test_datagen.flow_from_directory(
    test_data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# In[8]:


# Muat model ResNet50 yang sudah dilatih tanpa lapisan teratas
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)


# In[9]:


# Bekukan lapisan-lapisan model dasar
for layer in base_model.layers:
    layer.trainable = False


# In[10]:


# Bangun model
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])


# # Data Generating

# In[11]:


# Kompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[12]:


# Latih model
model.fit(train_dataset, epochs=50, validation_data=test_dataset)


# # Saving Model

# Vertex AI expects the model artifacts to be saved in `BASE_OUTPUT_DIRECTORY/model/` when you want to train a new version of a model

# In[12]:


saved_model_path = dataset_path + "/models/serena-emotion-detector/model"

# Do not uncomment this line, it will be done by setup.sh
model.save(saved_model_path)


# After saving, use `evaluate.ipynb` to evaluate the model after loading the artifacts.
