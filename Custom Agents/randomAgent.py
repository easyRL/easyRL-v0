from Agents import modelFreeAgent
import random

class CustomAgent(modelFreeAgent.ModelFreeAgent):
    displayName = 'Random Agent'

    def choose_action(self, state):
        self.time_steps += 1
        return random.randrange(self.action_size)

    def remember(self, state, action, reward, new_state, done):
        pass

    def reset(self):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def memsave(self):
        pass

    def memload(self, mem):
        pass

    def __deepcopy__(self, memodict={}):
        pass