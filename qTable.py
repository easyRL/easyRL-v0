from abc import ABC

import modelFreeAgent

class QTable(modelFreeAgent.ModelFreeAgent, ABC):
    displayName = 'Q Table'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Alpha', 0.00, 1.00, 0.01, 0.18, True, True)]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(QTable.newParameters)
        super().__init__(*args[:-paramLen])
        (self.alpha,) = args[-paramLen:]
        self.qtable = {}

    def getQvalue(self, state, action):
        return self.qtable.get((state, action), 0.0)

    def reset(self):
        self.qtable.clear()
