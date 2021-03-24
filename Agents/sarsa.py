from Agents.Collections import qTable


class sarsa(qTable.QTable):
    displayName = 'SARSA'

    def __init__(self, *args):
        super().__init__(*args)
        self.last_state = None
        self.last_action = None

    def remember(self, state, action, reward, new_state, _, done=False):
        # SARSA requires two timesteps of history. Since by default we arent given this, we must skip one to do so
        loss = 0
        if self.last_state is not None and self.last_action is not None:
            prevQValue = self.getQvalue(self.last_state, self.last_action)
            newQValue = self.getQvalue(state, action)
            if done:
                target = reward
            else:
                target = reward + self.gamma * newQValue
            loss = target - prevQValue
            self.qtable[(self.last_state, self.last_action)] = prevQValue + self.alpha * loss

        if done:
            self.last_state = None
            self.last_action = None
        else:
            self.last_state = state
            self.last_action = action
        return loss**2

    def update(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass
