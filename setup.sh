jupyter nbconvert ./trainer/task.ipynb --to python
sed -i 's/~\/gcs/\/gcs/g' ./trainer/task.py

python setup.py sdist --formats=gztar

gsutil cp ./dist/emotion-detector-module-0.1.tar.gz gs://serena-shsw-datasets/zip