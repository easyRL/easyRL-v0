import qTable

class sarsa(qTable.QTable):
    def __init__(self):
        super().__init__()

    def remember(self, state, action, reward, new_state):
        super().remember(state, action, reward, new_state)

    def update(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass