from Agents import modelFreeAgent
import numpy as np
from collections import deque
import random
import joblib
import cffi
import os
import pathlib

class DoubleDuelingQNative(modelFreeAgent.ModelFreeAgent):
    displayName = 'Double, Dueling Deep Q Native'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True, "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True, "The distance in timesteps between target model updates")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DoubleDuelingQNative.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, self.memory_size, self.target_update_interval = [int(arg) for arg in args[-paramLen:]]

        self.ffi = cffi.FFI()
        oldwd = pathlib.Path().absolute()
        curDir = oldwd / "../Agents/Native"
        os.chdir(curDir.as_posix())
        headerName = curDir / "deepQNative.h"
        with open(headerName) as headerFile:
            self.ffi.cdef(headerFile.read())

        self.ffi.set_source(
            "_deepQNative",
            """
            #include "deepQNative.h"
            """,
            libraries=["deepQNative"],
            library_dirs=[curDir.as_posix()],
            include_dirs=[curDir.as_posix()]
        )

        self.ffi.compile(verbose=True, tmpdir=curDir)

        import Agents.Native._deepQNative as _deepQNative
        self.nativeInterface = _deepQNative.lib
        self.nativeDQN = self.nativeInterface.createDQNc(self.state_size[0], self.action_size, self.gamma, self.batch_size, self.memory_size, self.target_update_interval)
        os.chdir(oldwd.as_posix())

    def __del__(self):
        self.nativeInterface.freeDQN(self.nativeDQN)

    def choose_action(self, state):
        cState = self.ffi.new("float[]", state.tolist())
        action = self.nativeInterface.chooseActionc(self.nativeDQN, cState)
        return action

    def remember(self, state, action, reward, new_state, done=False):
        cState = self.ffi.new("float[]", state.tolist())
        #cNewState = self.ffi.new("float[]", new_state)

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