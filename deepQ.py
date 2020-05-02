import modelFreeAgent

class DeepQ(modelFreeAgent.ModelFreeAgent):
    def __init__(self):
        super().__init__()
        self.model = None
        self.memory = None

    def choose_action(self, state):
        pass

    def remember(self, state, action, reward, new_state):
        super().remember(state, action, reward, new_state)

    def update(self):
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass