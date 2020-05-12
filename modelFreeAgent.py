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
            tkinter.Label(self, text='Max Epsilon: ').grid(row=2, column=0)
            self.maxEpsilon = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.maxEpsilon.set(1.0)
            self.maxEpsilon.grid(row=2, column=1)

            tkinter.Label(self, text='Min Epsilon: ').grid(row=3, column=0)
            self.minEpsilon = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.minEpsilon.set(0.1)
            self.minEpsilon.grid(row=3, column=1)

            tkinter.Label(self, text='Decay Rate: ').grid(row=4, column=0)
            self.decayRate = tkinter.Scale(self, from_=0.0, to=0.2, resolution=0.001, orient=tkinter.HORIZONTAL)
            self.decayRate.set(0.018)
            self.decayRate.grid(row=4, column=1)

        def getParameters(self):
            return super().getParameters() + (self.alpha.get(), self.maxEpsilon.get(), self.minEpsilon.get(), self.decayRate.get())

    @abstractmethod
    def remember(self, state, action, reward, new_state):
        pass

    @abstractmethod
    def reset(self):
        pass
