./setup.sh

gcloud ai custom-jobs create \
  --region=asia-southeast2 \
  --display-name=serena-emotion-detector \
  --python-package-uris=gs://serena-shsw-datasets/zip/emotion-detector-module-0.1.tar.gz \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,executor-image-uri=asia-docker.pkg.dev/vertex-ai/training/tf-gpu.2-9.py310:latest,python-module=trainer.task 
