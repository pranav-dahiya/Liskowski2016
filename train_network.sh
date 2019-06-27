#!/bin/sh

python generate_patches.py
python preprocessing_train.py
python preprocessing_test.py
python network.py
mv -v data/* Data-7/
