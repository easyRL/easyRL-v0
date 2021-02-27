import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from approximator import Approximator

from abc import ABC, abstractmethod
from torch.distributions import Categorical

class Policy(ABC):
    """
    Interface for a policy. Uses a function approximator to approximate
    the value of each action given a state and creates a probability
    distributions from those values to sample an action from.
    """
    def __init__(self, approximator):
        """
        Constructor for a policy.
        :param approximator: is the function approximator to use
        :type approximator: Agents.Policy.approximator.Approximator
        """
        if (not isinstance(approximator, Approximator)):
            raise ValueError("approximator must be an instance of Agents.Policy.approximator.Approximator.")
        self._approximator = approximator
    
    def count_params(self):
        """
        Counts the number of parameters used in this policy.
        :return: the number of parameters used in this policy
        :rtype: int
        """
        return self._approximator.count_params()
    
    def get_params(self):
        """
        Gets a list of the current parameters of this policy.
        :return: a list of the current parameters of this policy
        :rtype: list
        """
        return self._approximator.get_params()
    
    def set_params(self, params: np.ndarray):
        """
        Set the parameters of this policy to the ones given in the
        parameters as a numpy array. The length of the array must equal
        the number of parameters used by this policy.
        :param params: A numpy array of the parameters to set this
        policy to.
        :type params: numpy.ndarray
        """
        self._approximator.set_params(params)
    
    @abstractmethod
    def choose_action(self, state: np.ndarray):
        """
        Chooses an action by approximating the value of each action,
        creating a probability distribution from those values, and samples
        the action from that probability distribution.
        :param state: the state to choose an action for
        :type state: numpy.ndarray
        :return: the chosen action
        :rtype: int
        """
        pass
    
    @abstractmethod
    def log_prob(self, state: np.ndarray, action: int):
        """
        Computes the log-likelihood of taking the given action, given the
        state.
        :param state: the current state
        :type state: numpy.array
        :param action: the action being taken
        :type action: int
        :return: the log-likelihood of taking the action
        :rtype: float
        """
        pass

class CategoricalPolicy(Policy):
    """
    A categorical policy. Used for choosing from a range of actions.
    """
    def choose_action(self, state: tuple):
        """
        Chooses an action by approximating the value of each action,
        creating a probability distribution from those values, and samples
        the action from that probability distribution.
        :param state: the state to choose an action for
        :type state: numpy.ndarray
        :return: the chosen action
        :rtype: int
        """
        # Approximate the value of each action, convert results to tensor.
        values = self._approximator(state)
        values = torch.from_numpy(values).float()
        # Use softmax to determine the probability from each value.
        probs = F.softmax(values, dim=-1)
        # Create a categorical policy distribution from the probabilities.
        policy_dist = Categorical(probs)

        # Sample an action from the policy distribution.
        action = policy_dist.sample().item()
        
        # Return the chosen action.
        return action

    def log_prob(self, state: tuple, action):
        """
        Computes the log-likelihood of taking the given action, given the
        state.
        :param state: the current state
        :type state: numpy.array
        :param action: the action being taken
        :type action: int
        :return: the log-likelihood of taking the action
        :rtype: float
        """
        if (not isinstance(action, int) or action not in range(self._approximator.action_size)):
            raise ValueError("action must be an integer from the action space.")
        
        # Approximate the value of each action, convert results to tensor.
        values = self._approximator(state)
        values = torch.from_numpy(values).float()
        # Use softmax to determine the probability from each value.
        probs = F.softmax(values, dim=-1)
        # Create a categorical policy distribution from the probabilities.
        policy_dist = Categorical(probs)
        
        # Encapsulate action into a tensor.
        action = torch.tensor([action])
        # Calculate the log-likelihood of taking the given action.
        log_prob = policy_dist.log_prob(action).item()
        
        # Return the log-likelihood of taking the given action.
        return log_prob
