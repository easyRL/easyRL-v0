# RL-Framework
University of Washington Tacoma TCSS 556 Final Project
-------------
To run, first install the required pip packages using these commands in a Python 3.7 environment:
```
pip install pillow
pip install gym
pip install pandas
pip install numpy
pip install tensorflow
```

Then run the program:
```
python controller.py
```

We have CartPole with discretized states and a Q-Table, and CartPole with continuous states and Deep Q learning implemented.
We also have FrozenLake partially implemented with Q-Learning, which prints its output to the console instead of displaying in the GUI for now.
Also, FrozenLake's data is not yet displaying on the graph.

###For CartPole with discretized states and a Q-Table:

Click 'New Agent' -> 'Cart Pole Discretized' -> 'Q-Learning' Tab

###For CartPole with continuous states and a DQN:

Click 'New Agent' -> 'Cart Pole' -> 'Deep Q Learning' Tab

###For FrozenLake with a Q-Table:

Click 'New Agent' -> 'Frozen Lake' -> 'Q-Learning' Tab