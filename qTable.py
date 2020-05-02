import modelFreeAgent

class QTable(modelFreeAgent.ModelFreeAgent):
    def __init__(self):
        super().__init__()

    def choose_action(self, state):
        pass

    def reset(self):
        super().reset()