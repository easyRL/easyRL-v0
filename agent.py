from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass

    def __deepcopy__(self, memodict={}):
        pass
