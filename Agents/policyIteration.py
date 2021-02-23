from Agents import modelBasedAgent
from Agents import agent

class PolicyIteration(modelBasedAgent.ModelBasedAgent):
    newParameters = []
    parameters = modelBasedAgent.ModelBasedAgent.parameters + newParameters
    
    def __init__(self, *args):
        super().__init__(*args)
        self.value = []
        self.policy = []


    def update(self):
        super().update()

    def choose_action(self, state):
        super().choose_action(state)

    def __deepcopy__(self, memodict={}):
        pass
