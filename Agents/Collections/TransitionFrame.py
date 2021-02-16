class TransitionFrame:
    def __init__(self, state, action, reward, next_state, is_done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_done = is_done

class ActionTransitionFrame(TransitionFrame):
    def __init__(self, prev_action, state, action, reward, next_state, is_done):
        super().__init__(state, action, reward, next_state, is_done)
        self.prev_action = prev_action
