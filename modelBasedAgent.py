import tkinter
from abc import ABC

import self as self

import agent


class ModelBasedAgent(agent.Agent, ABC):
    class ParameterProfile(agent.Agent.ParameterProfile, ABC):
        def __init__(self, parent):
            super().__init__(parent)
            tkinter.Label(self, text='Theta: ').grid(row=1, column=0)
            self.theta = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.theta.set(0.97)
            self.theta.grid(row=1, column=1)

        # def getParameters(self):
        #     return (self.gamma.get(),)

        def getParameters(self):
            return super().getParameters() + (self.theta.get(),)
