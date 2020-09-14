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

class PPONative(modelFreeAgent.ModelFreeAgent):
    displayName = 'PPO Native'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Policy learning rate', 0.00001, 1, 0.00001, 0.001, True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Value learning rate', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Horizon', 10, 10000, 1, 50,
                                                             True, True,
                                                             "The number of timesteps over which the returns are calculated"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Epoch Size', 10, 100000, 1, 500,
                                                             True, True,
                                                             "The length of each epoch (likely should be the same as the max episode length)"),
                     modelFreeAgent.ModelFreeAgent.Parameter('PPO Epsilon', 0.00001, 0.5, 0.00001, 0.2,
                                                             True, True,
                                                             "A measure of how much a policy can change w.r.t. the states it's trained on"),
                     modelFreeAgent.ModelFreeAgent.Parameter('PPO Lambda', 0.5, 1, 0.001, 0.95,
                                                             True, True,
                                                             "A parameter that when set below 1, can decrease variance while maintaining reasonable bias")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(PPONative.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, _, _, self.horizon, self.epochSize, _, _ = [int(arg) for arg in args[-paramLen:]]
        _, self.policy_lr, self.value_lr, _, _, self.epsilon, self.lam = [arg for arg in args[-paramLen:]]

        oldwd = pathlib.Path().absolute()
        curDir = oldwd / "Agents/Native/ppoNative"
        os.chdir(curDir.as_posix())

        self.ffi = cffi.FFI()
        if platform.system() == "Windows":
            if not importlib.util.find_spec("Agents.Native.ppoNative.Release._ppoNative"):
                self.compileLib(curDir)
            import Agents.Native.ppoNative.Release._ppoNative as _ppoNative
        else:
            if not importlib.util.find_spec("Agents.Native.ppoNative._ppoNative"):
                self.compileLib(curDir)
            import Agents.Native.ppoNative._ppoNative as _ppoNative

        self.nativeInterface = _ppoNative.lib
        self.nativeppo = self.nativeInterface.createAgentc(self.state_size[0], self.action_size,
                                                                self.policy_lr, self.value_lr, self.gamma, self.horizon, self.epochSize,
                                                                self.batch_size, self.epsilon, self.lam)

        os.chdir(oldwd.as_posix())

    def compileLib(self, curDir):
        headerName = curDir / "ppoNative.h"
        outputDir = (curDir / "Release") if platform.system() == "Windows" else curDir
        with open(headerName) as headerFile:
            self.ffi.cdef(headerFile.read())
        self.ffi.set_source(
            "_ppoNative",
            """
            #include "ppoNative.h"
            """,
            libraries=["ppoNative"],
            library_dirs=[outputDir.as_posix()],
            include_dirs=[curDir.as_posix()]
        )
        self.ffi.compile(verbose=True, tmpdir=outputDir)

    def __del__(self):
        self.nativeInterface.freeAgentc(self.nativeppo)

    def choose_action(self, state):
        cState = self.ffi.new("float[]", list(state))
        action = self.nativeInterface.chooseActionc(self.nativeppo, cState)
        return action

    def remember(self, state, action, reward, new_state, done=False):
        cState = self.ffi.new("float[]", list(state))
        #cNewState = self.ffi.new("float[]", new_state)

        loss = self.nativeInterface.rememberc(self.nativeppo, cState, action, reward, done)
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