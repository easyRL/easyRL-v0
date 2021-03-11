import copy
import joblib
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agents import policyIteration
from Agents.Policy import approximator, policy
from collections import deque
from collections.abc import Iterable

class CEM(policyIteration.PolicyIteration):
    displayName = 'CEM'
    newParameters = [policyIteration.PolicyIteration.Parameter('Sigma', 0.001, 1.0, 0.001, 0.5, True, True, "The standard deviation of additive noise"),
                     policyIteration.PolicyIteration.Parameter('Population Size', 0, 100, 10, 10, True, True, "The size of the sample population"),
                     policyIteration.PolicyIteration.Parameter('Elite Fraction', 0.001, 1.0, 0.001, 0.2, True, True, "The proportion of the elite to consider for policy improvement.")]
    parameters = policyIteration.PolicyIteration.parameters + newParameters


    def __init__(self, *args):
        """
        Constructor for Cross Entropy Method agent.
        """
        paramLen = len(CEM.newParameters)
        super().__init__(*args[:-paramLen])
        self.sigma, self.pop_size, self.elite_frac = args[-paramLen:]
        # Convert the pop_size to an integer.
        self.pop_size = int(self.pop_size) 
        # Calculate the number of weight sets to consider the elite.
        self.elite = int(self.pop_size*self.elite_frac)
        
        '''
        Define the policy.
        '''
        # Create a deep learning approximator.
        approx = approximator.DeepApproximator(self.state_size, self.action_size, [16])
        # Create a categorical policy with a deep approximator for this agent.
        self._policy = policy.CategoricalPolicy(approx)
        # Weights of the policy
        self._best_weights = self.sigma*np.random.randn(self._policy.count_params())
        self._sample_policies = self._create_sample_policies()
        self._policy.set_params(self._best_weights)
        
    def choose_action(self, state, p: policy.Policy = None):
        """
        Chooses an action given the state and, if given, a policy. The
        policy p parameter is optional. If p is None, then the current
        policy of the agent will be used. Otherwise, the given policy p is
        used.
        :param state: is the current state of the environment
        :type state: numpy.ndarray
        :param policy: is the policy to use
        :type policy: Agents.Policy.policy.Policy
        :return: the chosen action
        :rtype: int
        """
        if (p is not None and not isinstance(p, policy.Policy)):
            raise ValueError("p must be a valid policy.Policy object.")
            
        # Initialize the action to -1.
        action = -1
        
        # Choose an action.
        if (p is None):
            # Choose an action using the current policy.
            action = self._policy.choose_action(state)
        else:
            # Choose an action using the given policy.
            action = p.choose_action(state)
        
        # Return the chosen action.
        return action

    def get_sample_policies(self):
        """
        Returns the current list of sample policies.
        :return: a list of the current sample policies
        :rtype: list
        """
        return self._sample_policies

    def update(self, trajectory: Iterable):
        """
        Updates the current policy given a the trajectory of the policy.
        :param trajectory: a list of transition frames from the episode.
        This represents the trajectory of the episode.
        :type trajectory: Iterable
        """
        if (not isinstance(trajectory, Iterable) or len(trajectory) != self.pop_size):
            raise ValueError("The length of the list of trajectories should be equal to the population size.")
        
        # Get the total episode rewards from each policy's trajectory.
        rewards = np.array([sum(transition.reward for transition in policy_t) for policy_t in trajectory])
        
        # Update the best weights based on the give rewards.
        elite_idxs = rewards.argsort()[-self.elite:]
        elite_weights = [self._sample_policies[i].get_params() for i in elite_idxs]
        self._best_weights = np.array(elite_weights).mean(axis=0)
        self._sample_policies = self._create_sample_policies()
        self._policy.set_params(self._best_weights)
    
    def _create_sample_policies(self):
        """
        Creates a list of sample policies. The length of the list is equal
        to the population size of this agent.
        :return: a list of sample policies
        :rtype: list
        """
        # An empty list to add the sample policies to.
        policies = []
        # Create n sample policies, where n is the population size.
        for i in range(self.pop_size):
            # Create a new policy that is a deep copy of the current policy.
            p = copy.deepcopy(self._policy)
            # Derive random weights from the current weights.
            sample_weights = self._best_weights + (self.sigma * np.random.randn(self._policy.count_params()))
            # Set the weights of the created policy to the derived ones.
            p.set_params(sample_weights)
            policies.append(p)
        # Return the created policies.
        return policies

    def save(self, filename):
        mem = self._policy.get_params()
        joblib.dump((CEM.displayName, mem), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != CEM.displayName:
            print('load failed')
        else:
            self._policy.set_params(mem)

    def memsave(self):
        return self._policy.get_params()

    def memload(self, mem):
        self._policy.set_params(mem)