#!/bin/bash

# An installation script for Easy-RL that gets around all the errors, tested in Ubuntu 20.04
# By Robert Cordingly

sudo apt update
sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y software-properties-common git python3.7 python3-pip python3.7-tk

python3.7 -m pip install pip --upgrade
python3.7 -m pip install pillow gym pandas numpy tensorflow joblib ttkthemes ttkwidgets opencv-python cffi gym[atari]
python3.7 -m pip install --upgrade --force-reinstall pillow

git clone https://github.com/easyRL/easyRL-v0

cd easyRL-v0
python3.7 EasyRL.py