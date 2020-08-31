# EasyRL-Framework

```
##Getting Started:

The following are basic instructions set up and run a copy of the
 application on your local machine for testing and development
  purposes. 
  
###Prerequisites:
  
To setup, run EasyRL.install. Then click OK to everything. 

##Running the Program:

Run EasyRL.exe.

##Running the tests
###The Framework:
The framework of the graphical interface is based on a tab system. 
Each tab represents an agent/environment pair which may be
trained/tested concurrently with, and independent of, all other tabs. 
 
Associated with each agent are parameters that are editable by the
user prior to training/testing. 
 
In order to control the training and testing processes, the user is
given the options to start/halt training/testing, save the trained
agent, load a trained agent , reset the currently loaded agent, 
and to save the results of the training/testing process.

The types of agent types available by default are as follows: 
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
The default agent types are all based on model-free algorithms
, and serve to approximate the optimal Q-function Q*. 
```
 ?^? (s) = arg (maxQ/a)^?(s, a)
```
###Selecting training parameters:

####For CartPole with discretized states and a Q-Table:
```
Click 'CartPole' -> 'Q-Learning' -> Click Set Model
```
####For CartPole with continuous states and a DQN:
```
Click 'CartPole' -> 'Deep Q Learning' -> Click Set Model
```
####For FrozenLake with a Q-Table:
```
Click 'Frozen Lake' -> 'Q-Learning' -> Click Set Model
```
###Demo
Our demonstration video makes it clear how a GUI can greatly simplify
 the process of defining, training, and test- ing a reinforcement
  learning agent. In the video, one can see how a user on a typical
   consumer-grade desktop computer can quickly produce RL agents with
    minimal knowledge of AI and even programming in general.

* [Demo](https://www.overleaf.com/read/vvwxqwghryqz
) - Video explanation of the RL-Framework application

##Reference on RL-Framework
###Built with:
See prerequisetes for packages used, no other programs in particular were used to create
EasyRL.

###Literature:
* [Paper](https://www.overleaf.com/read/vvwxqwghryqz
) - User Friendly Reinforcement Learning

###Contributing:
Notes on our code of conduct and or on the process for submitting
 pull requests from us.

###Authors:
* James Haines-Temons
* Neil Hulbert
* Kevin Flora
* Brandon Francis
* Sam Spillers
* Ken Gil Romero
* Benjamin De Jager
* Sam Wong
* Kevin Flora
* Bowei Huang
* Athirai A. Irissappane

###License:
This project is licensed by The University of Washington Tacoma

###Acknowledgements:
None

 
