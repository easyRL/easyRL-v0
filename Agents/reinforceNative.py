from Agents import modelFreeAgent
import numpy as np
from collections import deque
import random
import joblib
import cffi
import os
import pathlib
import platform
import importlib

class ReinforceNative(modelFreeAgent.ModelFreeAgent):
    displayName = 'Reinforce Native'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Policy learning rate', 0.00001, 1, 0.00001, 0.001, True, True,
                                                             "A learning rate that the Adam optimizer starts at")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(ReinforceNative.newParameters)
        super().__init__(*args[:-paramLen])
        (self.policy_lr,) = [arg for arg in args[-paramLen:]]

        oldwd = pathlib.Path().absolute()
        curDir = oldwd / "Agents/Native/reinforceNative"
        os.chdir(curDir.as_posix())

        self.ffi = cffi.FFI()
        if platform.system() == "Windows":
            if not importlib.util.find_spec("Agents.Native.reinforceNative.Release._reinforceNative"):
                self.compileLib(curDir)
            import Agents.Native.reinforceNative.Release._reinforceNative as _reinforceNative
        else:
            if not importlib.util.find_spec("Agents.Native.reinforceNative._reinforceNative"):
                self.compileLib(curDir)
            import Agents.Native.reinforceNative._reinforceNative as _reinforceNative

        self.nativeInterface = _reinforceNative.lib
        self.nativeReinforce = self.nativeInterface.createAgentc(self.state_size[0], self.action_size,
                                                                self.policy_lr, self.gamma)

        os.chdir(oldwd.as_posix())

    def compileLib(self, curDir):
        headerName = curDir / "reinforce.h"
        outputDir = (curDir / "Release") if platform.system() == "Windows" else curDir
        with open(headerName) as headerFile:
            self.ffi.cdef(headerFile.read())
        self.ffi.set_source(
            "_reinforceNative",
            """
            #include "reinforce.h"
            """,
            libraries=["reinforceNative"],
            library_dirs=[outputDir.as_posix()],
            include_dirs=[curDir.as_posix()]
        )
        self.ffi.compile(verbose=True, tmpdir=outputDir)

    def __del__(self):
        self.nativeInterface.freeAgentc(self.nativeReinforce)

    def choose_action(self, state):
        cState = self.ffi.new("float[]", list(state))
        action = self.nativeInterface.chooseActionc(self.nativeReinforce, cState)
        return action

    def remember(self, state, action, reward, new_state, done=False):
        cState = self.ffi.new("float[]", list(state))
        #cNewState = self.ffi.new("float[]", new_state)

        loss = self.nativeInterface.rememberc(self.nativeReinforce, cState, action, reward, done)
        return loss

    def update(self):
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def memsave(self):
        pass
    def memload(self, mem):
        pass