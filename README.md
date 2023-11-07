# How to Train in Vertex AI
1. Run setup.sh
2. Go to Vertex AI "Training" menu
3. Set Region from dropdown to "asia-southeast2 (Jakarta)"
4. Click "TRAIN NEW MODEL"
5. For "Training method", select "Custom training (advanced)", then click "Continue"
6. For "Model details", click "Train new model" if training a new model or "Train new version" if training a new version for a model
7. Fill the appropiate fields, click next
8. For "Training container", click "Pre-built container" and set Model framework and version to the one you are using in your code (e.g. TensorFlow 2.12, check your improts)
9. In "Pre-built container settings" for Package location, browse to "serena-shsw-datasets/zip/" and select "emotion-detector-module-0.1.tar.gz"
10. For "Python module" field, set to "trainer.task"
11. Ignore "Model output directory"
12. Click continue
13. Click continue on "Hyperparameters" menu
14. For "Compute and pricing", set region to "asia-southeast2 (Jakarta)"
15. For "Machine type" it depends on the hyperparameters your keras model uses (i.e. batch_size, steps_per_epoch, epochs, etc.). From testing, machine type "n1-highcpu-32" can handle batch_size = 128 and probably more. The more batch_size, the faster training will be, but the more vCPUs & memories you will need. This is a trial and error process since you need to adjust your hyperaparameters based on your dataset samples & epochs. But the default quota for vCPUs in asia-southeast2 is 42, so you should only be able to set machine type with 32 vCPUs as your strongest vm.
16. Set "Disk type" to SSD & 50GB disk size
17. Click continue
18. Click "START TRAINING"
19. Your training job should show in the Vertex AI Training menu on that region. Click on the name and click "View logs" to see the training progress