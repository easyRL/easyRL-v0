import agent

class ModelFreeAgent(agent.Agent):
    def __init__(self):
        super().__init__()

    def remember(self, state, action, reward, new_state):
        pass

    def reset(self):
        pass