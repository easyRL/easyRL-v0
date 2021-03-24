# EasyRL-Framework - Desktop and Cloud Application

# Installation for Windows x64 (releases):

----> Download the latest installer executable from https://github.com/easyRL/easyRL-v0/releases/

----> Run installer executable and follow directions

----> See https://youtu.be/Sc5phA_TR_o for EasyRL functionalities

----> See https://youtu.be/Jaa9TG3-F2c for EasyRL Cloud functionalities

# Building from source for Windows/MacOS/Linux (master branch):

For easy installation onto Ubuntu 20.04 the install.sh script can be used to automatically download all the dependencies.

```
./install.sh
```

If you are manually setting up the environment install the following dependencies:

To setup, first install the required pip packages using these commands
in a Python 3.7 environment:

For Ubuntu 20.04 install the following packages. 

```
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
```

Using Pip, install the follow Pyton dependencies:

```
python3.7 -m pip install pillow
python3.7 -m pip install gym
python3.7 -m pip install gym[atari]
python3.7 -m pip install atari-py
python3.7 -m pip install pandas
python3.7 -m pip install numpy
python3.7 -m pip install tensorflow
python3.7 -m pip install joblib
python3.7 -m pip install ttkthemes
python3.7 -m pip install ttkwidgets
python3.7 -m pip install opencv-python
python3.7 -m pip install cffi
python3.7 -m pip install torch
python3.7 -m pip install botocore
python3.7 -m pip install boto3
python3.7 -m pip install paramiko
python3.7 -m pip install django
python3.7 -m pip install django-storages
python3.7 -m pip install --upgrade --force-reinstall pillow
```

(if not on Windows):

```
pip (or pip3) install gym[atari]
```

(OR if on Windows with the Visual C++ Build Tools installed):

```
pip install atari-py
```

Download the appropriate LibTorch C++ Release package from https://pytorch.org/ .

Ensure that the Microsoft Visual C++ compiler is installed.

Inside Agents/Native directory, run

```
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .
```

and build from the build files generated in Agents/Native using the MSVC compiler.

# Running the Program:

If installed from the installer, run EasyRL.exe

If built from source, run for the desktop GUI application.

```
python3.7 EasyRL.py
```

To run the application in terminal view mode add the terminal flag:
```
python3.7 EasyRL.py --terminal
```

To run the application in cloud mode (where data is saved locally and submitted to S3) add keys and a jobID:
```
python3.7 EasyRL.py --terminal --secretKey {KEY} --accessKey {KEY} --jobID {UUID} --sessionToken {OPTIONAL}
```

From the root project directory

# Other Depdendencies
<details>
  <summary>Click to expand...</summary>
  
  Here are the required dependencies:
  
  ```
  -- visual c++ installation
  -- compile native agents (cmake list)
  -- absl-py==0.9.0
  -- astor==0.8.1
  -- atari-py==0.2.6
  -- cachetools==4.0.0
  -- certifi==2019.11.28
  -- chardet==3.0.4
  -- cloudpickle==1.2.2
  -- cycler==0.10.0
  -- decorator==4.4.1
  -- future==0.18.2
  -- gast==0.2.2
  -- google-auth==1.11.0
  -- google-auth-oauthlib==0.4.1
  -- google-pasta==0.1.8
  -- grpcio==1.27.1
  -- gym~=0.17.2
  -- h5py==2.10.0
  -- idna==2.8
  -- imageio==2.6.1
  -- joblib~=0.16.0
  -- Keras==2.3.1
  -- Keras-Applications==1.0.8
  -- Keras-Preprocessing==1.1.0
  -- kiwisolver==1.1.0
  -- lxml==4.5.0
  -- Markdown==3.2
  -- matplotlib==3.1.3
  -- networkx==2.4
  -- numpy~=1.19.0
  -- oauthlib==3.1.0
  -- opencv-python~=4.3.0.36
  -- opt-einsum==3.1.0
  -- pandas==1.0.1
  -- Pillow~=7.2.0
  -- protobuf==3.11.3
  -- pyasn1==0.4.8
  -- pyasn1-modules==0.2.8
  -- pyglet==1.2.4
  -- pyparsing==2.4.6
  -- python-dateutil==2.8.1
  -- pytils==0.3
  -- pytz==2019.3
  -- PyWavelets==1.1.1
  -- PyYAML==5.3
  -- requests==2.22.0
  -- requests-oauthlib==1.3.0
  -- rsa==4.0
  -- scikit-image==0.16.2
  -- scipy==1.4.1
  -- six==1.14.0
  -- tensorboard==2.1.0
  -- tensorboardX==2.0
  -- tensorflow==2.1.0
  -- tensorflow-estimator==2.1.0
  -- termcolor==1.1.0
  -- tf==1.0.0
  -- tools==0.1.9
  -- torchvision==0.6.0
  -- ttkthemes~=3.1.0
  -- ttkwidgets==0.11.0
  -- urllib3==1.25.8
  -- Werkzeug==1.0.0
  -- wrapt==1.11.2
  -- xlrd==1.2.0
  -- XlsxWriter==1.2.9
  -- xlutils==2.0.0
  -- xlwt==1.3.0
  -- interval~=1.0.0
  ```
 </details>

# Types of inbuild agents:
<details>
  <summary>Click to expand...</summary>
  Currently available agents:
  
  ```
  Q-Table SARSA/Q-Learning
  deep Q-learning
  deep recurrent Q-learning
  action deep recurrent Q-learning
  double, dueling deep q native
  drqn native
  conv drqn native
  ppo native
  reinforce native
  actorcritic native
  trpo
  sac
  npg
  ddpg
  cem
  ```
</details>

# Setting Up EasyRL Cloud:

The first step is to deploy the AWS Lambda function. This can be easily done by configuring the AWS CLI and running the publish.sh script. For an interactive tutorial for installing the AWS CLI and all dependencies needed for SAAF simply run:

```
./lambda/install.sh
```

Now that the AWS CLI has been configured, modify /lambda/python_template/deploy/config.json and enter a valid AWS ARN for the lambda security role. Then run:

```
# Configure AWS CLI to deploy lambda functions to us-east-1. Skip if used install script.
aws configure

# Add AWS ARN to config.json for deployment in lambdaRoleARN value.
nano lambda/python_template/deploy/config.json

# Run publish script to automatically deploy and test function.
./lambda/python_template/deploy/publish.sh 1 0 0 0 512
```

This will deploy the function as cloudBridge. The lambda function defined the AWS AMI used for the backend, modify /lambda/python_template/scr/handler.py if you would like to use a different AMI or Github repository for the backend. Once the function is deployed the front end must be deployed to any webserver with both django, python, boto3, and the AWS CLI configured to call the lambda function. This should be all the configuration needed. Source code for the web server and GUI is available in the separate EasyRL-Web repository.

# How to Run EasyRL Cloud

This application uses Django as a webserver to host the interface. To run the server you must have Python with the following packages installed:

- django
- django storage
- boto3
- botocore

```
python3.7 -m pip install django boto3 botocore django-storages
```

Before starting the server, we need to set up the environment variables for the WebGUI (The key must be the same as the one using AWS Lambda)

```
export AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=xXxXXxxXxXxxXXXxXxxXXxxXXXxxxXxXxxxxxxX
```

Make migration and migrate project relative to the model modification (required only when the code in models.py changed)

```
python3.7 manage.py makemigrations
python3.7 manage.py migrate
```

Once setup, start the server by running manage.py This will default to port 8000:

```
python3.7 manage.py runserver [optional port number]
```

Once the server is running, open the webpage below in your browser and log in with AWS credentials to begin!

```
http://SERVER_IP:SERVER_PORT/easyRL_app/
```

![](./webpage/login.png)

After successfully logging into the system using the credentials, the webpage redirects to the main dashboard as follow:

![](./webpage/app.png)


### EasyRL Contributors:

* Neil Hulbert
* James Haines-Temons
* Brandon Francis
* Sam Spillers
* Ken Gil Romero
* Benjamin De Jager
* Sam Wong
* Bowei Huang
* Kevin Flora
* Athirai A. Irissappane

### EasyRL Cloud Contributors:

* Robert Cordingly
* Tucker Stewart
* Varik Hoang
* Rashad Hatchett
* Maham Rashid
* Nazim Zerrouki
* Shrustishree Sumanth
* Egor Maksimenka
