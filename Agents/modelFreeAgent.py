from abc import ABC, abstractmethod

from Agents import agent

"""This is an abstract model-free agent class that allows a user to define
their own custom agent by extending this class as a class named 'CustomAgent'.
"""
class ModelFreeAgent(agent.Agent, ABC):
    displayName = 'Model Free Agent'
    newParameters = [agent.Agent.Parameter('Min Epsilon', 0.00, 1.00, 0.01, 0.1, True, True, "The minimum value of epsilon during training; the minimum probability that the model will select a random action over its desired one"),
                     agent.Agent.Parameter('Max Epsilon', 0.00, 1.00, 0.01, 1.0, True, True, "The maximum value of epsilon during training; the maximum/starting probability that the model will select a random action over its desired one"),
                     agent.Agent.Parameter('Decay Rate', 0.00, 0.20, 0.001, 0.018, True, True, "The amount to decrease epsilon by each timestep")]
    parameters = agent.Agent.parameters + newParameters

    def __init__(self, *args):
        """Constructor method
        :param args: the parameters associated with the agent
        :type args: tuple
        """
        paramLen = len(ModelFreeAgent.newParameters)
        super().__init__(*args[:-paramLen])
        self.min_epsilon, self.max_epsilon, self.decay_rate = args[-paramLen:]
    
    def apply_hindsight(self):
        pass

    @abstractmethod
    def remember(self, state, action, reward, new_state, done):
        """'Remembers' the state and action taken during an episode
        :param state: the original state of the environment
        :param action: the action the agent took in the environment
        :param reward: the reward the agent observed given its action
        :type reward: number
        :param new_state: the new state that the agent found itself after taking the action
        :param episode: the episode number
        :type episode: int
        :param done: whether the episode was finished after taking the action
        :type done: bool
        :return: the MSE loss for the predicted q-values
        :rtype: number
        """
        pass

    @abstractmethod
    def reset(self):
        """Resets the agent to its original state, removing the results of any training
        :return: None
        :rtype: None
        """
        pass
