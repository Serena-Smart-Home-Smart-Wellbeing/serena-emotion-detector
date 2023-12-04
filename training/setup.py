from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'imbalanced-learn==0.11.0',
    'opencv-python==4.8.1.78',
    'matplotlib==3.8.2',
    'pydot==1.4.2'
]

setup(
    name='emotion-detector-module',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Serena Emotion Detector',
    install_requires=REQUIRED_PACKAGES
)