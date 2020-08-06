Header = "Welcome to Portal RL - A user friendly environment for \n" \
         "learning and deploying reinforcement learning environments and agents\n\n\n"

Section1 =  "  MAIN WORK AREA (Tab X) \n\n"  \
            "    - Getting Started: \n\n" \
            "      1) Number of episodes -\n" \
            "         This parameter is used to define how many times the program should run the same \n" \
            "         simulation. Depending on the complexity of the model, a user might choose to run the simulation\n" \
            "         thousands of times to learn an optimal policy for the task it is trying to perform.\n" \
            "         NEW USER TIP:\n" \
            "         We recommend starting with a low number of episodes and gradually increasing the amount\n" \
            "         This will give you a sense of how reinforcement learning works and the affect\n" \
            "         the number of episodes has on various agent/environment interactions.\n\n" \
            "      2) Max Steps -\n" \
            "         This number dictates how many actions will be taken during each episode before a training\n" \
            "         is concluded\n" \
            "         NEW USER TIP:\n" \
            "         Arbitrarily large numbers of steps don't always work out to better learning. Some of the\n" \
            "         suggested starting environments have a limited size. As you become more familiar with the \n" \
            "         environments you will get a feel for how many steps it should take to find an optimal policy\n" \
            "         and can adjust accordingly. You will learn that some reinforcement learning programs take a \n" \
            "         large number of episodes (which can include multiple days of training) and an ideal number of\n" \
            "         steps will allow an agent to run episodes efficiently.\n\n" \
            "      3) Select Environment - \n" \
            "         This drop down menu will allow you to select a reinforcement learning environment\n" \
            "         from the available python gym sandboxes including a selection of atari games\n" \
            "         NEW USER TIP:\n" \
            "         We recommend starting with CartPole, CartPole Discrete, FrozenLake, Pendulum, Acrobat, or \n" \
            "         MountainCar. As you become more in tune with how these environments work move into the atari\n" \
            "         games, and if you feel inspired we hope you explore the internet for ways to develop your own\n" \
            "         environments and interact with our API (Found in our Advanced Options section). \n\n" \
            "      4) Select Agent - \n" \
            "         This drop down menu contains several of the most studied reinforcement learning \n" \
            "         algorithms to match to your environment. \n" \
            "         NEW USER TIP: \n" \
            "         Don't worry, we have installed some guard rails to keep you from matching agents and \n" \
            "         environments that don't work together. As you study how reinforcement learning works you will \n" \
            "         understand why those combinations don't work together.\n\n" \
            "      5) Set Model -\n" \
            "         Once you have selected an environment and an agent this will open the training interface.\n\n"

Section2 =  "     - Training Interface - \n\n" \
            "       1) Gamma - \n" \
            "          This value (0 <= X <= 1) represents the discount that the reinforcement learning agents assign to \n" \
            "          future rewards when calculating the value of a being between one action and the next. How this is\n" \
            "          done varies from algorithm to algorithm.\n" \
            "          NEW USER TIP:\n" \
            "          This value is, in almost all circumstances, very close to one.\n\n" \
            "       2) Min/Max Epsilon and Decay Rate -\n" \
            "          This value represents the proportion of time an agent should choose a random action or consult\n" \
            "          its policy. A value of 1 means that it will always choose a random action and a value of 0 means it\n" \
            "          will always consult the policy. This is set to the Max value initially and decrements at\n" \
            "          intervals by the decay rate during training. When testing these should be set to zero.\n\n" \
            "       3) Batch Size -\n" \
            "          When training a neural network this is the size of the group of actions and rewards that the \n" \
            "          network will consider each time while training its decision process. In any neural network this is\n" \
            "          a value that is fine tuned through testing.\n\n" \
            "       4) Memory size -\n" \
            "          This is the maximimum number of state, action, and reward tuples that are stored for reference\n" \
            "          by the agent.\n\n" \
            "       5) History Length -\n" \
            "          Neural networks often rely on information that is gained from its perception of the environment, and\n" \
            "          in environments that a single image doesn't tell the user much about what is happening in a given\n" \
            "          state this variable determines the number of chronological frames that a neural network will \n" \
            "          consider when updating its policy.\n\n" \
            "       6) Alpha -\n" \
            "          Alpha is another name for the learning rate or step size. It defines the impact new information has\n" \
            "          when overwriting previous values.\n\n" \
            "       7) Train -\n" \
            "          Locks in current parameters and initiates training of the agent. Will display statistical\n" \
            "          information about the training process in the readout space.\n\n" \
            "       8) Halt -\n" \
            "          Prematurely ends the current session of training or testing.\n\n" \
            "       9) Test -\n" \
            "          Set the agent to perform in the environment exclusively according to the current policy and \n" \
            "          returns results. NOTE: It is advised that the user sets epsilon to zero during testing at this \n" \
            "          time. It will occasionally produce a bug otherwise. This will be fixed in a future patch.\n\n" \
            "      10) Save Agent -\n" \
            "          Saves the state of the current agent to a file.\n\n" \
            "      11) Load Agent -\n" \
            "          Opens a file selection window that allows the user to select an agent to be loaded. The agent\n" \
            "          must match the Agent/Environment combination selected during the Start Screen of the tab.\n\n" \
            "      12) Reset -\n" \
            "          Sets the parameters and the agent state to the default.\n\n" \
            "      13) Save Results -\n" \
            "          Opens a file selection window and allows the user to write a save file containing the results of\n" \
            "          training or testing an agent/environment interaction as a csv file."

Section3 =  "   TABS AND BUTTONS AND VISUALS\n\n" \
            "     1) Tabs - \n" \
            "        Each tab is a new thread that works on a different agent/environment combination. Add new tabs by \n" \
            "        clicking the plus button.\n\n" \
            "     2) Close Current Tab -\n" \
            "        This will end the thread being run by the tab and close the tab.\n\n" \
            "     3) Reset Current Tab -\n" \
            "        Ends the thread and sets the tab to its opening default state.\n\n" \
            "     4) Load Environment -\n" \
            "        Opens a file selection window that allows the user to load a custom built environment into the set \n" \
            "        of environments in the dropdown menu.\n\n" \
            "     5) Load Agent -\n" \
            "        Opens a file selection window and allows the user to load a custom built agent in the set of agents\n" \
            "        in the dropdown menu.\n\n" \
            "     6) Legend -\n" \
            "        MSE - Mean squared error for recording the loss of an agent/environment interaction\n" \
            "        Episode Reward - The resulting reward achieved during an episode\n" \
            "        Epsilon - The current epsilon value during training\n" \
            "        These contribute to a readout of the performance of an agent/environment interaction.\n\n"

API_info =  "   Advanced - API information: \n\n" \
            "     The Portal API requires methods for environments and agents as follows -\n\n" \
            "     - Environment: \n" \
            "       Must extend the environment abstract class and contain all abstract methods within and as set forth\n" \
            "       in the API documentation.\n\n" \
            "     - Agent:\n" \
            "       Must extend the abstract modelBasedAgent or modelFreeAgent class from the Agents library or one of\n" \
            "       their child classes and contain all methods therein described by that class and the abstract Agent\n" \
            "       class and as set forth in the API documentation.\n\n"

display_episode_speed = "the 'display episode speed' parameter can be changed mid-simulation to adjust how quickly the rendering runs. Keep in mind\nthat each loop is showing the most recent episode, so if you set it high it will often repeat the same episode and if set\nlow it will often skip episodes."

def getHelpGettingStarted():

    return Header + Section1 + Section2 + Section3 + API_info + display_episode_speed


# getHelpText()
