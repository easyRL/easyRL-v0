from Agents import modelBasedAgent


class ValueIteration(modelBasedAgent.ModelBasedAgent):
    def __init__(self):
        super().__init__()
        self.value = []

    def update(self):
        super().update()

    def choose_action(self, state):
        super().choose_action(state)

    def __deepcopy__(self, memodict={}):
        pass
