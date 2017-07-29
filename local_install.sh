#!/bin/bash

rm -rf  build
rm -rf  keras_utilities.egg-info
rm -rf  dist

python setup.py sdist bdist_wheel

pip install keras_utilities --no-index --find-links=dist --force-reinstall --no-deps -U