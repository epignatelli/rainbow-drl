!#bin/bash

# create conda environment
conda create -n rainbow python=3.7
source activate rainbow


# install jaxlib
PYTHON_VERSION=cp37  # alternatives: cp36, cp37, cp38
CUDA_VERSION=cuda100  # alternatives: cuda100, cuda101, cuda102, cuda110
PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'

# install jaxlib
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.51-$PYTHON_VERSION-none-$PLATFORM.whl

# install jax
pip install --upgrade jax

# install remaining packages
pip install -r requirements.txt