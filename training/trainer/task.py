#!/usr/bin/env python
# coding: utf-8

# # Serena Emotion Detector - Training Notebook
# 
# This notebook is used to setup and train Serena Emotion Detector in Vertex AI. The output will be saved to our GCS bucket `serena-shsw-datasets/models` folder.  
# To evaluate the model, use `evaluate.ipynb` notebook in this directory.
# 

# ## Background
# 
# Serena Emotion Detector is a CNN model that detects 7 emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) from a person's front-facing photo. We use [FER2013](https://www.kaggle.com/deadskull7/fer2013) dataset since it is a popular dataset for emotion detection.
# 
# When we were starting with creating our model, we used to create the architecture from scratch. But after multiple trial and errors, the best we could get was around 64% accuracy. Even then, that took about 3 hours for every 10 epoch training session in Vertex AI. Other than that, our own models always faced problems where it would classify wrong emotions or would just be biased towards one emotion class.
# 
# After learning from our mistakes, learning more about CNN, and learning from tutorials; we decided to use transfer learning. We used transfer learning since it would be quicker to train and we don't have to design the architecture from scratch, instead we would just need to modify the dataset and some layers to fit our needs of detecting 7 classes of emotions.
# 
# The model we use for transfer learning is [MobileNetV2](https://www.ict-srilanka.com/blog/what-is-mobilenetv2). We use MobileNetV2 since it is designed to be lightweight for devices with limited resources (e.g. mobile phones, IoT devices, etc). We think this would make predictions quicker when deployed on Cloud Run or later in the future when we want to embed it directly into our IoT device `SerenBox`.
# 

# ## Setup
# 
# We store our dataset in GCS. There are 7 classes, each class seperated into a folder. We'll save the model using the latest `.keras` instead of `.h5` since it's more modern and easier to move around since it will save the weight and model configuration in one file.  
# If you want to try it out yourself, you need to replace `train_dataset_path` to your own FER-2013 dataset path. You can download FER-2013 dataset [here](https://www.kaggle.com/msambare/fer2013).
# 
# > 🚧 Warning
# >
# > This notebook was designed to be run in OUR Vertex AI environment. If you want to run it yourself, you need to change some code to fit your environment.  
# > You can directly use our model without having to train it first by following the steps in `evaluate.ipynb` notebook.
# 

# In[1]:


import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import Model, layers
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint

gcs_path = "/gcs/serena-shsw-datasets/"
train_dataset_path = os.path.join(
    gcs_path, "FER-2013/train"  # TODO: change this to your own dataset
)
model_save_path = os.path.join(
    gcs_path,
    "models/serena-emotion-detector.keras",  # TODO: change this to your own path
)
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ## Processing Training Data
# 

# Read 40% of the images from each folder, convert them into numpy array, then append them into `training_data`. We only read 40% of the images since we keep running into memory limit errors even when using `n1-highmem-32` VM + 2 `NVIDIA_TESLA_T4` accelerators. We think this is due reshaping each image to 224x224 which is pretty big. But even with 40% of the data, we could still get accuracy of more than 90% for the model.
# 

# In[1]:


# There are 28709 images in the training set for FER-2013
train_dataset_total = 28709
sample_size = train_dataset_total * 0.40
print("Sample size: ", sample_size)

training_data = []
img_size = 224
img_array = []


def stratified_sample_size_for_class(
    stratum_size, train_dataset_total=train_dataset_total, sample_size=sample_size
):
    return round(((sample_size / train_dataset_total) * stratum_size))


# Use this if memory is limited and you are getting sigkill errors
def create_training_data_stratified_sample():
    for category in classes:
        path = os.path.join(train_dataset_path, category)
        class_num = classes.index(category)
        stratum_size = len(os.listdir(path))
        sample_size = stratified_sample_size_for_class(stratum_size)
        print(
            "Class: ",
            category,
            "Stratum size: ",
            stratum_size,
            "Sample size: ",
            sample_size,
        )
        for img in os.listdir(path)[:sample_size]:
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


# In[6]:


create_training_data_stratified_sample()


# In[7]:


print("Total training data size: ", len(training_data))


# Randomize the training data to avoid bias
# 

# In[8]:


import random


random.shuffle(training_data)


# Reshape training data to fit our model
# 

# In[9]:


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
Y = np.array(y)

print(X.shape)
print(Y.shape)


# Normalize the data
# 

# In[10]:


X = X / 255.0


# ## Creating Transfer Learning Model
# 

# Create pretrained model from `MobileNetV2`.
# 

# In[11]:


pretrained_model = MobileNetV2()
pretrained_model.summary()


# Create new layers from the pretrained model.
# 

# In[12]:


input_layer = pretrained_model.layers[0].input
base_output_layer = pretrained_model.layers[-2].output

output_layer = layers.Dense(128)(base_output_layer)
output_layer = layers.Activation("relu")(output_layer)
output_layer = layers.Dense(64)(output_layer)
output_layer = layers.Activation("relu")(output_layer)
output_layer = layers.Dense(7, activation="softmax")(output_layer)

print(output_layer)

new_model = Model(
    inputs=input_layer,
    outputs=output_layer,
)
new_model._name = "serena-emotion-detector"
new_model.summary()


# In[13]:


new_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)


# Start training the model and saving the best model.
# 
# > 🚧 Warning
# >
# > DO NOT TRAIN DIRECTLY ON YOUR LOCAL COMPUTER, unless you have a really beefy computer with atleast 100GB of RAM. Why? Because the dataset is huge and it would take a loooonngggg time to train locally.
# > To train, run `train.sh` to package this notebook and train it on Vertex AI using `n1-highmem-8` VM + 1 `NVIDIA_TESLA_T4` accelerator.
# 

# In[ ]:


new_model.fit(X, Y, epochs=25)

new_model.save(model_save_path)

