import qTable


class QLearning(qTable.QTable):
    displayName = 'Q Learning'

    def __init__(self, input_size, action_size, learning_rate, gamma):
        super().__init__(action_size, learning_rate, gamma)

    def remember(self, state, action, reward, new_state, done=False):
        prevQValue = self.getQvalue(state, action)
        newQValue = self.getQvalue(new_state, self.choose_action(new_state))
        if done:
            target = -reward
        else:
            target = reward + self.gamma * newQValue
        loss = target - prevQValue
        self.qtable[(state, action)] = prevQValue + self.learning_rate * loss
        return loss**2

    def choose_action(self, state):
        q = [self.getQvalue(state, a) for a in range(self.action_size)]
        maxQ = max(q)
        return q.index(maxQ)

    def __deepcopy__(self, memodict={}):
        pass
