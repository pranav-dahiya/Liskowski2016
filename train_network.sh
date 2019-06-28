#!/bin/sh

python metrics.py > /dev/null 2>&1 &
python generate_patches.py
python preprocessing_train.py
python preprocessing_test.py
python network.py
mkdir Data-9
mv -v data/* Data-9/
