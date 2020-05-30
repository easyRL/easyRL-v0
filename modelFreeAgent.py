import tkinter
from abc import ABC, abstractmethod

import agent


class ModelFreeAgent(agent.Agent, ABC):
    displayName = 'Model Free Agent'

    parameters = []
    parameters.append(agent.Agent.Parameter('Alpha', 0.00, 1.00, 0.01, 0.18, True, True))
    parameters.append(agent.Agent.Parameter('Max Epsilon', 0.00, 1.00, 0.01, 1.0, True, True))
    parameters.append(agent.Agent.Parameter('Min Epsilon', 0.00, 1.00, 0.01, 0.1, True, True))
    parameters.append(agent.Agent.Parameter('Decay Rate', 0.00, 0.20, 0.001, 0.018, True, True))
    parameters = agent.Agent.parameters + parameters

    @abstractmethod
    def remember(self, state, action, reward, new_state):
        pass

    @abstractmethod
    def reset(self):
        pass
