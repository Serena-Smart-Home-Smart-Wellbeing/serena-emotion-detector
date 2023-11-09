jupyter nbconvert ./trainer/task.ipynb --to python

# Change some code to be runable
sed -i 's/~\/gcs/\/gcs/g' ./trainer/task.py
sed -i 's/\/home\/jupyter\/gcs/\/gcs/g' ./trainer/task.py
sed -i '/get_ipython/d' ./trainer/task.py

# Zip and upload to GCS
python setup.py sdist --formats=gztar
gsutil cp ./dist/emotion-detector-module-0.1.tar.gz gs://serena-shsw-datasets/zip
