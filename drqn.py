import deepQ

class DRQN(deepQ.DeepQ):
    def __init__(self):
        super().__init__()

    def choose_action(self, state):
        super().choose_action(state)

    def update(self):
        super().update()

    def reset(self):
        super().reset()