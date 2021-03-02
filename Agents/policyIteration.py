from Agents import modelBasedAgent
from abc import ABC, abstractmethod
from collections.abc import Iterable

class PolicyIteration(modelBasedAgent.ModelBasedAgent, ABC):
    displayName = 'Policy Iteration Method'
    newParameters = []
    parameters = modelBasedAgent.ModelBasedAgent.parameters + newParameters
    
    def __init__(self, *args):
        super().__init__(*args)
        self._policy = None
    
    @abstractmethod
    def update(self, trajectory: Iterable):
        """
        Updates the current policy given a the trajectory of the policy.
        :param trajectory: a list of transition frames from the episode.
        This represents the trajectory of the episode.
        :type trajectory: Iterable
        """
        pass
