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

# Run this `gcsfuse` cell if you are using Vertex AI workbench and can't list the folders inside of "/gcs"

# In[1]:




# In[2]:


import os

import cv2
import numpy as np
from tensorflow import keras, data
from keras import layers, Model
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint

gcs_path = "/gcs/serena-shsw-datasets/"
train_dataset_path = os.path.join(
    gcs_path, "FER-2013/train"  # TODO: change this to your own dataset
)
test_dataset_path = os.path.join(
    gcs_path, "FER-2013/test"  # TODO: change this to your own dataset
)
model_save_path = os.path.join(
    gcs_path,
    "models/serena-emotion-detector.keras",  # TODO: change this to your own path
)

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ## Processing Training Data
# 

# img_size = 224img_size = 224img_size = 224Read 40% of the images from each folder, convert them into numpy array, then append them into `training_data`. We only read 40% of the images since we keep running into memory limit errors even when using `n1-highmem-32` VM + 2 `NVIDIA_TESLA_T4` accelerators. We think this is due reshaping each image to 224x224 which is pretty big. But even with 40% of the data, we could still get accuracy of around 94% for the model.
# 

# Load training & validation data

# In[4]:


# Kwargs for image_dataset_from_directory
img_size = 224
labels = "inferred"
label_mode = "int"
class_names = classes
color_mode = "rgb"
batch_size = 32 * 7
image_size = (img_size, img_size)
shuffle = True
interpolation = "bilinear"
follow_links = False


def create_training_data():
    training_data = keras.utils.image_dataset_from_directory(
        directory=train_dataset_path,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        interpolation=interpolation,
        follow_links=follow_links,
        seed=123,
    )

    return training_data


def create_validation_data():
    validation_data = keras.utils.image_dataset_from_directory(
        directory=test_dataset_path,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        interpolation=interpolation,
        follow_links=follow_links,
        seed=321,
    )

    return validation_data


# In[5]:


training_data = create_training_data()


# In[6]:


validation_data = create_validation_data()


# In[7]:


print(training_data.class_names)
print(validation_data.class_names)
print("Same order: ", training_data.class_names == validation_data.class_names)


# Setup image loading strategy

# In[8]:


AUTOTUNE = data.AUTOTUNE

training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)


# Normalize dataset to range 0-1

# In[9]:


normalization_layer = layers.Rescaling(1./255)

normalized_training_data = training_data.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_training_data))
first_image = image_batch[0]

print("min: ",np.min(first_image), "max: ",np.max(first_image))


# ## Creating Transfer Learning Model
# 

# Create pretrained model from `MobileNetV2`.
# 

# In[10]:


pretrained_model = MobileNetV2()
pretrained_model.summary()


# Create new layers from the pretrained model.
# 

# In[11]:


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
new_model.summary()


# In[12]:


new_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)


# ## Train Model

# Start training the model and saving the best model.
# 
# > 🚧 Warning
# >
# > DO NOT TRAIN DIRECTLY ON YOUR LOCAL COMPUTER, unless you have a really beefy computer with atleast 100GB of RAM. Why? Because the dataset is huge and it would take a loooonngggg time to train locally.
# > To train, run `train.sh` to package this notebook and train it on Vertex AI using `n1-highmem-8` VM + 1 `NVIDIA_TESLA_T4` accelerator.
# 

# In[ ]:


history = new_model.fit(normalized_training_data, validation_data=validation_data, epochs=2)


# In[ ]:


new_model.save(model_save_path)
print("Saved model to: " + model_save_path)


# ## Evaluate Training Results
# 

# In[ ]:


import matplotlib.pyplot as plt
# Mendapatkan data pelatihan (training) dari history
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']

# Membuat grafik untuk loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Membuat grafik untuk accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Menampilkan grafik
plt.tight_layout()
plt.savefig(fname=gcs_path + "plots/serena-emotion-detector.png")
plt.show()


# In[ ]:




