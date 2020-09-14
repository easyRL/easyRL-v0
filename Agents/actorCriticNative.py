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

class ActorCriticNative(modelFreeAgent.ModelFreeAgent):
    displayName = 'ActorCritic Native'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Policy learning rate', 0.00001, 1, 0.00001, 0.001, True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Value learning rate', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Horizon', 10, 10000, 1, 50,
                                                             True, True,
                                                             "The number of timesteps over which the returns are calculated"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Epoch Size', 10, 100000, 1, 500,
                                                             True, True,
                                                             "The length of each epoch (likely should be the same as the max episode length)")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(ActorCriticNative.newParameters)
        super().__init__(*args[:-paramLen])
        _, _, self.horizon, self.epochSize = [int(arg) for arg in args[-paramLen:]]
        self.policy_lr, self.value_lr, _, _ = [arg for arg in args[-paramLen:]]

        oldwd = pathlib.Path().absolute()
        curDir = oldwd / "Agents/Native/actorCriticNative"
        os.chdir(curDir.as_posix())

        self.ffi = cffi.FFI()
        if platform.system() == "Windows":
            if not importlib.util.find_spec("Agents.Native.actorCriticNative.Release._actorCriticNative"):
                self.compileLib(curDir)
            import Agents.Native.actorCriticNative.Release._actorCriticNative as _actorCriticNative
        else:
            if not importlib.util.find_spec("Agents.Native.actorCriticNative._actorCriticNative"):
                self.compileLib(curDir)
            import Agents.Native.actorCriticNative._actorCriticNative as _actorCriticNative

        self.nativeInterface = _actorCriticNative.lib
        self.nativeActorCritic = self.nativeInterface.createAgentc(self.state_size[0], self.action_size,
                                                                self.policy_lr, self.value_lr, self.gamma, self.horizon, self.epochSize)

        os.chdir(oldwd.as_posix())

    def compileLib(self, curDir):
        headerName = curDir / "actorCritic.h"
        outputDir = (curDir / "Release") if platform.system() == "Windows" else curDir
        with open(headerName) as headerFile:
            self.ffi.cdef(headerFile.read())
        self.ffi.set_source(
            "_actorCriticNative",
            """
            #include "actorCritic.h"
            """,
            libraries=["acNative"],
            library_dirs=[outputDir.as_posix()],
            include_dirs=[curDir.as_posix()]
        )
        self.ffi.compile(verbose=True, tmpdir=outputDir)

    def __del__(self):
        self.nativeInterface.freeAgentc(self.nativeActorCritic)

    def choose_action(self, state):
        cState = self.ffi.new("float[]", list(state))
        action = self.nativeInterface.chooseActionc(self.nativeActorCritic, cState)
        return action

    def remember(self, state, action, reward, new_state, done=False):
        cState = self.ffi.new("float[]", list(state))
        #cNewState = self.ffi.new("float[]", new_state)

        loss = self.nativeInterface.rememberc(self.nativeActorCritic, cState, action, reward, done)
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