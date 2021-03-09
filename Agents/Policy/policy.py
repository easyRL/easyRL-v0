import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Agents.Policy.approximator import Approximator

from abc import ABC, abstractmethod
from torch.distributions import Categorical

class Policy(ABC):
    """
    Interface for a policy. Uses a function approximator to approximate
    the value of each action given a state and creates a probability
    distributions from those values to sample an action from.
    
    Adapted from 'https://github.com/zafarali/policy-gradient-methods/
    blob/f0d83a80ddc772dcad0c851aac9bfd41d436c274/pg_methods/policies.py'.
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
    
    def get_params(self, flatten: bool = True):
        """
        Gets the parameters used by this approximator. The parameters
        for this model is the weights and bias of each layer. The
        parameters are returned as a one-dimensional numpy array if flatten
        is true which is by default, otherwise the parameters are return
        in the format of the model being used.
        :param flatten: whether to flatten the parameters to a
        one-dimensional array or not
        :type param: bool
        :return: the parameters used by this approximator
        """
        return self._approximator.get_params(flatten)
    
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
    
    def zero_grad(self):
        """
        Zeros out the gradient of the approximator.
        """
        self._approximator.zero_grad()
    
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
    def get_distribution(self, states: np.ndarray, detach: bool = True):
        """
        Creates a policy distribution given an array of states.
        :param states: an array of states to create the policy distribution.
        :type states: np.ndarray
        :param detach: determines whether to detach the result from the
        tensor or not. Set to True as default.
        :type detach: bool
        :return: the probability distribution of this policy.
        :rtype: torch.distribution
        """
        pass
    
    @abstractmethod
    def logit(self, state: np.ndarray, action: int):
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
    def choose_action(self, state: np.ndarray, detach: bool = True):
        """
        Chooses an action by approximating the value of each action,
        creating a probability distribution from those values, and samples
        the action from that probability distribution.
        :param state: the state to choose an action for
        :type state: numpy.ndarray
        :param detach: determines whether to detach the result from the
        tensor or not. Set to True as default.
        :type detach: bool
        :return: the chosen action
        :rtype: torch.Tensor or int
        """
        # Approximate the value of each action, convert results to tensor.
        values = self._approximator(state)
        # Use softmax to determine the probability from each value.
        probs = F.softmax(values, dim=-1)
        # Create a categorical policy distribution from the probabilities.
        policy_dist = Categorical(probs)

        # Sample an action from the policy distribution.
        action = policy_dist.sample()
        
        # If detach is true, then detach the result from the tensor.
        if (detach):
            action = action.item()
        
        # Return the chosen action.
        return action
    
    def get_distribution(self, states: np.ndarray, detach: bool = True):
        """
        Creates a policy distribution given an array of states.
        :param states: an array of states to create the policy distribution.
        :type states: np.ndarray
        :param detach: determines whether to detach the result from the
        tensor or not. Set to True as default.
        :type detach: bool
        :return: the probability distribution of this policy.
        :rtype: torch.distribution
        """
        if (not isinstance(states, np.ndarray) or states.shape[1:] != self._approximator.state_size):
            raise ValueError("states must be a numpy array with each state having the shape {}.".format(self.state_size))
        
        # Approximate the value of each action for each state given.
        values = []
        for state in states:
            approx_values = self._approximator(state)
            values.append(approx_values)
        values = torch.stack(values)
        # Use softmax to determine the probability from each value.
        probs = F.softmax(values, dim=-1)
        
        # If detach is true, then detach the result from the tensor.
        if detach:
            probs = probs.detach()
        
        # Create and a categorical policy distribution from the probabilities.
        policy_dist = Categorical(probs)
        return policy_dist

    def logit(self, state: np.ndarray, action: int, detach: bool = True):
        """
        Computes the log-likelihood of taking the given action, given the
        state.
        :param state: the current state
        :type state: numpy.ndarray
        :param action: the action being taken
        :type action: int
        :param detach: determines whether to detach the result from the
        tensor or not. Set to True as default.
        :type detach: bool
        :return: the log-likelihood of taking the action
        :rtype: torch.Tensor or float
        """
        if (not isinstance(action, int) or action not in range(self._approximator.action_size)):
            raise ValueError("action must be an integer from the action space.")
        
        # Approximate the value of each action, convert results to tensor.
        values = self._approximator(state)
        # Use softmax to determine the probability from each value.
        probs = F.softmax(values, dim=-1)
        # Create a categorical policy distribution from the probabilities.
        policy_dist = Categorical(probs)
        
        # Encapsulate action into a tensor.
        action = torch.tensor([action])
        # Calculate the log-likelihood of taking the given action.
        logit = policy_dist.log_prob(action)
        
        # If detach is true, then detach the result from the tensor.
        if (detach):
            logit = logit.item()
        
        # Return the log-likelihood of taking the given action.
        return logit
