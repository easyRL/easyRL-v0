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
sudo apt install -y unzip

sudo apt install -y gcc 
sudo apt install -y g++
sudo apt install -y cmake
sudo apt install -y python3.7-dev
sudo apt install -y libxml2-dev 
sudo apt install -y libxslt1-dev
sudo apt install -y build-essential 
sudo apt install -y libssl-dev 
sudo apt install -y libffi-dev

sudo python3.7 -m pip install pip --upgrade
sudo python3.7 -m pip install pillow
sudo python3.7 -m pip install gym
sudo python3.7 -m pip install pandas
sudo python3.7 -m pip install numpy
sudo python3.7 -m pip install tensorflow
sudo python3.7 -m pip install joblib
sudo python3.7 -m pip install ttkthemes
sudo python3.7 -m pip install ttkwidgets
sudo python3.7 -m pip install opencv-python
sudo python3.7 -m pip install cffi
sudo python3.7 -m pip install gym[atari]
sudo python3.7 -m pip install boto3
sudo python3.7 -m pip install torch
sudo python3.7 -m pip install --upgrade --force-reinstall pillow

wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.8.0%2Bcpu.zip
rm libtorch-shared-with-deps-1.8.0+cpu.zip
cd easyRL-v0/Agents/Native
cmake -DCMAKE_PREFIX_PATH=~/libtorch .
make
cd ~


#git clone https://github.com/easyRL/easyRL-v0

#cd easyRL-v0
#python3.7 EasyRL.py