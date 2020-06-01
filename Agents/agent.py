from abc import ABC, abstractmethod


class Agent(ABC):
    class Parameter():
        def __init__(self, name, min, max, resolution, default, hasSlider, hasTextInput):
            self.name = name
            self.min = min
            self.max = max
            self.resolution = resolution
            self.default = default
            self.hasSlider = hasSlider
            self.hasTextInput = hasTextInput

    parameters = [Parameter('Gamma', 0.00, 1.00, 0.01, 0.97, True, True)]

    def __init__(self, state_size, action_size, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.time_steps = 0

    @abstractmethod
    def choose_action(self, state):
        self.time_steps += 1

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def memsave(self):
        pass

    @abstractmethod
    def memload(self, mem):
        pass

    def __deepcopy__(self, memodict={}):
        pass
