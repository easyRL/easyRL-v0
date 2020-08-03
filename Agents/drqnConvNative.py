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

class DRQNConvNative(modelFreeAgent.ModelFreeAgent):
    displayName = 'Conv DRQN Native'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True, "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True, "The distance in timesteps between target model updates"),
                     modelFreeAgent.ModelFreeAgent.Parameter('History Length', 0, 20, 1, 10, True, True, "The number of recent timesteps to use as input")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DRQNConvNative.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, self.memory_size, self.target_update_interval, self.historyLength = [int(arg) for arg in args[-paramLen:]]

        oldwd = pathlib.Path().absolute()
        curDir = oldwd / "Agents/Native/drqnConvNative"
        os.chdir(curDir.as_posix())

        self.ffi = cffi.FFI()
        if platform.system() == "Windows":
            if not importlib.util.find_spec("Agents.Native.drqnConvNative.Release._drqnConvNative"):
                self.compileLib(curDir)
            import Agents.Native.drqnConvNative.Release._drqnConvNative as _drqnConvNative
        else:
            if not importlib.util.find_spec("Agents.Native.drqnConvNative._drqnConvNative"):
                self.compileLib(curDir)
            import Agents.Native.drqnConvNative._drqnConvNative as _drqnConvNative

        self.nativeInterface = _drqnConvNative.lib
        self.nativeDRQNConv = self.nativeInterface.createAgentc(self.state_size[2], self.state_size[0], self.state_size[1], self.action_size,
                                                                self.gamma,
                                                                self.batch_size, self.memory_size,
                                                                self.target_update_interval, self.historyLength)

        os.chdir(oldwd.as_posix())

    def compileLib(self, curDir):
        headerName = curDir / "drqnConvNative.h"
        outputDir = (curDir / "Release") if platform.system() == "Windows" else curDir
        with open(headerName) as headerFile:
            self.ffi.cdef(headerFile.read())
        self.ffi.set_source(
            "_drqnConvNative",
            """
            #include "drqnConvNative.h"
            """,
            libraries=["drqnConvNative"],
            library_dirs=[outputDir.as_posix()],
            include_dirs=[curDir.as_posix()]
        )
        self.ffi.compile(verbose=True, tmpdir=outputDir)

    def __del__(self):
        self.nativeInterface.freeAgentc(self.nativeDRQNConv)

    def choose_action(self, state):
        cState = self.ffi.new("float[]", state.flatten().tolist())
        action = self.nativeInterface.chooseActionc(self.nativeDRQNConv, cState)
        return action

    def remember(self, state, action, reward, new_state, done=False):
        cState = self.ffi.new("float[]", state.flatten().tolist())
        #cNewState = self.ffi.new("float[]", new_state)

        loss = self.nativeInterface.rememberc(self.nativeDRQNConv, cState, action, reward, done)
        return loss

    def update(self):
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass

    def save(self, filename):
        cFilename = self.ffi.new("char[]", filename.encode('ascii'))
        self.nativeInterface.savec(self.nativeDRQNConv, cFilename)

    def load(self, filename):
        cFilename = self.ffi.new("char[]", filename.encode('ascii'))
        self.nativeInterface.loadc(self.nativeDRQNConv, cFilename)

    def memsave(self):
        return self.nativeInterface.memsavec(self.nativeDRQNConv)

    def memload(self, mem):
        self.nativeInterface.memloadc(self.nativeDRQNConv, mem)