import tkinter
from abc import ABC, abstractmethod

from paramFrame import ParamFrame


class Agent(ABC):
    def __init__(self, state_size, action_size, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.time_steps = 0
        pass

    @abstractmethod
    def choose_action(self, state):
        self.time_steps += 1
        pass

    def __deepcopy__(self, memodict={}):
        pass

    class ParameterProfile(ParamFrame, ABC):
        def __init__(self, parent):
            super().__init__(parent)
            tkinter.Label(self, text='Gamma: ').grid(row=0, column=0)
            self.gamma = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.gamma.set(0.97)
            self.gamma.grid(row=0, column=1)

        def getParameters(self):
            return (self.gamma.get(),)
