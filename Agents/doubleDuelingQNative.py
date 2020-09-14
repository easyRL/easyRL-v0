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

class DoubleDuelingQNative(modelFreeAgent.ModelFreeAgent):
    displayName = 'Double, Dueling Deep Q Native'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True, "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True, "The distance in timesteps between target model updates"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Learning Rate', 0.00001, 100, 0.00001, 0.001, True, True, "The rate at which the parameters respond to environment observations")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DoubleDuelingQNative.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, self.memory_size, self.target_update_interval, _ = [int(arg) for arg in args[-paramLen:]]
        _, _, _, self.learning_rate = [arg for arg in args[-paramLen:]]

        oldwd = pathlib.Path().absolute()
        curDir = oldwd / "Agents/Native/deepQNative"
        os.chdir(curDir.as_posix())

        self.ffi = cffi.FFI()
        if platform.system() == "Windows":
            if not importlib.util.find_spec("Agents.Native.deepQNative.Release._deepQNative"):
                self.compileLib(curDir)
            import Agents.Native.deepQNative.Release._deepQNative as _deepQNative
        else:
            if not importlib.util.find_spec("Agents.Native.deepQNative._deepQNative"):
                self.compileLib(curDir)
            import Agents.Native.deepQNative._deepQNative as _deepQNative

        self.nativeInterface = _deepQNative.lib
        self.nativeDQN = self.nativeInterface.createAgentc(self.state_size[0], self.action_size, self.gamma, self.batch_size, self.memory_size, self.target_update_interval, self.learning_rate)

        os.chdir(oldwd.as_posix())


    def compileLib(self, curDir):
        headerName = curDir / "deepQNative.h"
        outputDir = (curDir / "Release") if platform.system() == "Windows" else curDir
        with open(headerName) as headerFile:
            self.ffi.cdef(headerFile.read())
        self.ffi.set_source(
            "_deepQNative",
            """
            #include "deepQNative.h"
            """,
            libraries=["deepQNative"],
            library_dirs=[outputDir.as_posix()],
            include_dirs=[curDir.as_posix()]
        )
        self.ffi.compile(verbose=True, tmpdir=outputDir)

    def __del__(self):
        self.nativeInterface.freeAgentc(self.nativeDQN)

    def choose_action(self, state):
        cState = self.ffi.new("float[]", list(state))
        action = self.nativeInterface.chooseActionc(self.nativeDQN, cState)
        return action

    def remember(self, state, action, reward, new_state, done=False):
        cState = self.ffi.new("float[]", list(state))
        #cNewState = self.ffi.new("float[]", new_state)

        done = 1 if done else 0

        loss = self.nativeInterface.rememberc(self.nativeDQN, cState, action, reward, done)
        return loss

    def update(self):
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass

    def save(self, filename):
        cFilename = self.ffi.new("char[]", filename.encode('ascii'))
        self.nativeInterface.savec(self.nativeDQN, cFilename)

    def load(self, filename):
        cFilename = self.ffi.new("char[]", filename.encode('ascii'))
        self.nativeInterface.loadc(self.nativeDQN, cFilename)

    def memsave(self):
        return self.nativeInterface.memsavec(self.nativeDQN)

    def memload(self, mem):
        self.nativeInterface.memloadc(self.nativeDQN, mem)