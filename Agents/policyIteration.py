from Agents import modelBasedAgent
from abc import ABC, abstractmethod

class PolicyIteration(modelBasedAgent.ModelBasedAgent, ABC):
    displayName = 'Policy Iteration Method'
    newParameters = []
    parameters = modelBasedAgent.ModelBasedAgent.parameters + newParameters
    
    def __init__(self, *args):
        super().__init__(*args)
        self._policy = None
    
    @abstractmethod
    def update(self, rewards):
        """
        Updates the current policy given an array of rewards.
        :param rewards: an array of rewards from the episode
        :type rewards: numpy.ndarray
        """
        pass
