#!/usr/bin/env python
# coding: utf-8

# # Datasets

# When using GCS buckets, use "/gcs" instead of "gs://"

# In[8]:


meow = "/gcs"
dataset_path = "/gcs/serena-shsw-datasets"
training_dataset = dataset_path + "/FER-2013/train"
test_dataset = dataset_path + "/FER-2013/test"

get_ipython().system('echo "Train"')
get_ipython().system('ls {training_dataset}')
get_ipython().system('echo "Test"')
get_ipython().system('ls {test_dataset}')


# # Saving Model

# In[ ]:


model.save(dataset_path + "/models/emotion-detector')

