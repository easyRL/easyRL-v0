import tkinter
from abc import ABC, abstractmethod

from paramFrame import ParamFrame

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

    def __init__(self):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass

    def __deepcopy__(self, memodict={}):
        pass
