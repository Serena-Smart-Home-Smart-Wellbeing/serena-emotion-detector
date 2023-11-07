jupyter nbconvert ./trainer/task.ipynb --to python

python setup.py sdist --formats=gztar

gsutil cp ./dist/emotion-detector-module-0.1.tar.gz gs://serena-shsw-datasets/zip