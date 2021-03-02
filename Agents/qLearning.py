from Agents.Collections import qTable
import joblib

class QLearning(qTable.QTable):
    displayName = 'Q Learning'

    def __init__(self, *args):
        super().__init__(*args)

    def remember(self, state, action, reward, new_state, done=False):
        prevQValue = self.getQvalue(state, action)
        newQValue = self.getQvalue(new_state, self.choose_action(new_state))
        if done:
            target = reward
        else:
            target = reward + self.gamma * newQValue
        loss = target - prevQValue
        self.qtable[(state, action)] = prevQValue + self.alpha * loss
        return loss**2

    def __deepcopy__(self, memodict={}):
        pass
