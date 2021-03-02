from abc import ABC, abstractmethod

"""This is an abstract model-free agent class that allows a user to define
their own custom agent by extending this class as a class named 'CustomAgent'.
"""
class Agent(ABC):
    """
    This is a parameter class that defines a parameter of an extended agent
    """
    class Parameter():
        def __init__(self, name, min, max, resolution, default, hasSlider, hasTextInput, toolTipText=""):
            self.name = name
            self.min = min
            self.max = max
            self.resolution = resolution
            self.default = default
            self.hasSlider = hasSlider
            self.hasTextInput = hasTextInput
            self.toolTipText = toolTipText

    parameters = [Parameter('Gamma', 0.00, 1.00, 0.001, 0.97, True, True, "The factor by which to discount future rewards")]

    def __init__(self, state_size, action_size, gamma):
        """The constructor method
        :param state_size: the shape of the environment state
        :type state_size: tuple
        :param action_size: the number of possible actions
        :type action_size: int
        :param gamma: the discount factor
        :type gamma: float
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.time_steps = 0
    
    def get_empty_state(self):
        """
        Gets the empty game state.
        :return: A representation of an empty game state.
        :rtype: list
        """
        shape = self.state_size
        if len(shape) >= 2:
            return [[[-10000]] * shape[0] for _ in range(shape[1])]
        return [-10000] * shape[0]

    @abstractmethod
    def choose_action(self, state):
        """Returns the action chosen by the agent's current policy given a state
        :param state: the current state of the environment
        :type state: tuple
        :return: the action chosen by the agent
        :rtype: int
        """
        self.time_steps += 1

    @abstractmethod
    def save(self, filename):
        """Saves the agent's Q-function to a given file location
        :param filename: the name of the file location to save the Q-function
        :type filename: str
        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    def load(self, filename):
        """Loads the agent's Q-function from a given file location
        :param filename: the name of the file location from which to load the Q-function
        :type filename: str
        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    def memsave(self):
        """Returns a representation of the agent's Q-function
        :return: a representation of the agent's Q-function
        """
        pass

    @abstractmethod
    def memload(self, mem):
        """Loads a passed Q-function
        :param mem: the Q-function to be loaded
        :return: None
        :rtype: None
        """
        pass

    def __deepcopy__(self, memodict={}):
        pass
