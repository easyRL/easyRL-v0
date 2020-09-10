# EasyRL-Framework

# Installation for Windows (releases):
  
----> open EasyRL folder

----> run setup.exe

----> click next

----> click next

----> click install

# Installation by building source from the scratch for Windows/IOS/Linux (master branch):
To setup, first install the required pip packages using these commands
 in a Python 3.7 environment:
```
pip install pillow
pip install gym
pip install pandas
pip install numpy
pip install tensorflow
pip install joblib
pip install ttkthemes
pip install ttkwidgets
pip install opencv-python
pip install cffi
```

(if not on Windows): 
```
pip (or pip3) install gym[atari]
```
(OR if on Windows with the Visual C++ Build Tools installed):
```
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
pip install git+https://github.com/Kojoley/atari-py.git
```

Then, download and extract the .zip file

Next is to download and extract RL-FrameworkRelease_win-x64_torch160_cpu.zip from https://github.com/nhulbert/RL-Framework/releases.
Copy and paste the RL-Framework/Agents/Native folder from RL-FrameworkRelease_win-x64_torch160_cpu.zip to your own RL-Framework/Agents/Native folder.

# Running the Program:
First, run the following on either Mac, Linux, or Windows PC...

-Run EasyRL.exe

OR

-Navigate to the RL-Framework folder then

-For Linux Command line:
```
./EasyRL
```
-For Windows CmdPrompt
```
python EasyRL.py
```

# GUI Example for training :
Click 'CartPole' -> 'Q-Learning' -> Click Set Model -> Click Train

# Command-Line Example for training:


# other dependensies requirements.txt/check ---? 
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


# Types of inbuild agents: 
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
actorcritic native. 
```

###Contributors (check):
* James Haines-Temons
* Neil Hulbert
* Kevin Flora
* Brandon Francis
* Sam Spillers
* Ken Gil Romero
* Benjamin De Jager
* Sam Wong
* Bowei Huang
* Athirai A. Irissappane
