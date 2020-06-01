from Agents import qTable
import joblib

class QLearning(qTable.QTable):
    displayName = 'Q Learning'

    def __init__(self, *args):
        super().__init__(*args)

    def remember(self, state, action, reward, new_state, _, done=False):
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

    def save(self, filename):
        joblib.dump((QLearning.displayName, self.qtable), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != QLearning.displayName:
            print('load failed')
        else:
            self.qtable = mem
            print('load successful')

    def memsave(self):
        return self.qtable

    def memload(self, mem):
        self.qtable = mem