from abc import ABC, abstractmethod

import agent


class ModelFreeAgent(agent.Agent, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def remember(self, state, action, reward, new_state):
        pass

    @abstractmethod
    def reset(self):
        pass
