# How to Train in Vertex AI
1. Develop your model in "/serena-emotion-detector/trainer/task.ipynb" notebook. DO NOT TRAIN IN THIS NOTEBOOK! Instead, when running cells containing the model.fit() line, only check if it runs (i.e. it outputs Epoch 1/10) and if it does, abort it.
2. When you are done, run setup.sh
3. Go to Vertex AI "Training" menu
4. Set Region from dropdown to "asia-southeast2 (Jakarta)"
5. Click "TRAIN NEW MODEL"
6. For "Training method", select "Custom training (advanced)", then click "Continue"
7. For "Model details", click "Train new model" if training a new model or "Train new version" if training a new version for a model
8. Fill the appropiate fields, click next
9. For "Training container", click "Pre-built container" and set Model framework and version to the one you are using in your code (e.g. TensorFlow 2.12, check your improts)
10. In "Pre-built container settings" for Package location, browse to "serena-shsw-datasets/zip/" and select "emotion-detector-module-0.1.tar.gz"
11. For "Python module" field, set to "trainer.task"
12. Ignore "Model output directory"
13. Click continue
14. Click continue on "Hyperparameters" menu
15. For "Compute and pricing", set region to "asia-southeast2 (Jakarta)"
16. For "Machine type" it depends on the hyperparameters your keras model uses (i.e. batch_size, steps_per_epoch, epochs, etc.). From testing, machine type "n1-highcpu-32" can handle batch_size = 128 and probably more. So far, the best combo seems to be batch_size=128 + n1-highcpu-32 which resulted in training time of 3h36m and accuracy of 0.41. The more batch_size, the faster training will be, but the more vCPUs & memories you will need and it could reduce accuracy due to overfitting. This is a trial and error process since you need to adjust your hyperaparameters based on your dataset samples & epochs. But the default quota for vCPUs in asia-southeast2 is 42, so you should only be able to set machine type with 32 vCPUs as your strongest vm. Adding GPU accelerators could improve speed, but you need to request quota for it.
17. Set "Disk type" to SSD & 100GB disk size
18. Click continue
19. Click "START TRAINING"
20. Your training job should show in the Vertex AI Training menu on that region. Click on the name and click "View logs" to see the training progress

# How to Evaluate Model
Since training won't be done in notebook due to the time it takes, evaluating can't happen in the same notebook. Instead, do these steps after training:
1. Create a new notebook file, let's call it "evaluate.ipynb"
2. Load your model artifacts in the notebook using function like `model.load(PATH_TO_ARTIFACT)`. Your artifact could be .h5/SavedModel/.keras, we use SavedModel (saved_model.pb) file here which is stored in GCS so replace your path to it in the GCS bucket.
3. Import matplotlib, pandas, and any other evaluating libraries you need to run your evaluations