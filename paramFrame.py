import tkinter
from abc import ABC, abstractmethod


class ParamFrame(tkinter.Frame, ABC):
    @abstractmethod
    def getParameters(self):
        pass
