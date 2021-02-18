#!/bin/bash

# An installation script for Easy-RL that gets around all the errors, tested in Ubuntu 20.04
# By Robert Cordingly

sudo apt update
sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y software-properties-common 
sudo apt install -y git 
sudo apt install -y python3.7 
sudo apt install -y python3-pip 
sudo apt install -y python3.7-tk 
sudo apt install -y ffmpeg 
sudo apt install -y libsm6 
sudo apt install -y libxext6

python3.7 -m pip install pip --upgrade
python3.7 -m pip install pillow
python3.7 -m pip install gym
python3.7 -m pip install pandas
python3.7 -m pip install numpy
python3.7 -m pip install tensorflow
python3.7 -m pip install joblib
python3.7 -m pip install ttkthemes
python3.7 -m pip install ttkwidgets
python3.7 -m pip install opencv-python
python3.7 -m pip install cffi
python3.7 -m pip install gym[atari]
python3.7 -m pip install --upgrade --force-reinstall pillow

#git clone https://github.com/easyRL/easyRL-v0

#cd easyRL-v0
#python3.7 EasyRL.py