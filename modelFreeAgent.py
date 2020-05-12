import tkinter
from abc import ABC, abstractmethod

import agent


class ModelFreeAgent(agent.Agent, ABC):
    class ParameterProfile(agent.Agent.ParameterProfile, ABC):
        def __init__(self, parent):
            super().__init__(parent)
            # alpha
            tkinter.Label(self, text='Alpha: ').grid(row=1, column=0)
            self.alpha = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.alpha.set(0.97)
            self.alpha.grid(row=1, column=1)

            # epsilon
            tkinter.Label(self, text='Epsilon: ').grid(row=2, column=0)
            self.epsilon = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.epsilon.set(0.97)
            self.epsilon.grid(row=2, column=1)

        def getParameters(self):
            return super().getParameters() + (self.alpha.get(), self.epsilon.get())

    @abstractmethod
    def remember(self, state, action, reward, new_state):
        pass

    @abstractmethod
    def reset(self):
        pass
