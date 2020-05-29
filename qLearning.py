import qTable


class QLearning(qTable.QTable):
    def __init__(self, state_size, output_size, gamma, alpha, min_epsilon, max_epsilon, decay_rate):
        super().__init__(state_size, output_size, gamma, alpha, min_epsilon, max_epsilon, decay_rate)

    def remember(self, state, action, reward, new_state):
        prevQValue = self.getQvalue(state, action)
        newQValue = self.getQvalue(new_state, self.choose_action(new_state))
        loss = reward + self.gamma * newQValue - prevQValue
        self.qtable[(state, action)] = prevQValue + self.alpha * loss
        return loss**2

    def choose_action(self, state):
        q = [self.getQvalue(state, a) for a in range(self.action_size)]
        maxQ = max(q)
        return q.index(maxQ)

    def __deepcopy__(self, memodict={}):
        pass
