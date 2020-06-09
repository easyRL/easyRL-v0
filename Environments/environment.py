import gym
from abc import ABC

"""This is an abstract environment class that allows a user to define
their own custom environment by extending this class as a 'CustomEnv' class.
"""
class Environment(ABC):
    displayName = 'Environment'

    """Constructor method
    """
    def __init__(self):
        self.action_size = None
        self.state_size = None
        self.state = None
        self.done = None

    def step(self, action):
        """Advances the state of the environment one time step given the agent's action
        :param action: the action the agent will take before taking the step
        :type action: int
        :return: the reward the agent obtains by taking the action and the time step advancing
        :rtype: number
        """
        pass

    def reset(self):
        """Resets the environment to an initial state
        :return: the state of the reset environment
        :rtype: tuple
        """
        pass

    def sample_action(self):
        """Samples an action from the environment
        :return: some action the agent can take in the environment
        :rtype: int
        """
        pass

    def render(self):
        """Renders the environment as an image
        :return: an image representing the current environment state
        :rtype: PIL.Image
        """
        pass

    def close(self):
        """Closes the environment, freeing any resources it is using
        :return: None
        :rtype: None
        """
        pass