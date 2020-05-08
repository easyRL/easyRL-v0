import random
import numpy as np

class Model:
    def __init__(self):
        # these can be set directly from the Controller based on user input from the View
        self.environment = None
        self.agent = None
        self.isHalted = False

    def run_learning(self, messageQueue, total_episodes, learning_rate, max_steps, gamma, max_epsilon, min_epsilon, decay_rate):
        epsilon = max_epsilon

        for episode in range(total_episodes):
            self.environment.reset()

            for step in range(max_steps):
                old_state = self.environment.state
                exp_exp_tradeoff = random.uniform(0, 1)

                if exp_exp_tradeoff > epsilon:
                    action = self.agent.choose_action(old_state)
                else:
                    action = self.environment.sample_action()

                message = Model.Message(self.environment.render())
                messageQueue.put(message)

                reward = self.environment.step(action)

                self.agent.remember(old_state, action, reward, self.environment.state, learning_rate, gamma)

                if self.environment.done or self.isHalted:
                    break

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            #print(step, epsilon)
            if self.isHalted:
                self.isHalted = False
                break
        message = Model.Message(None, True)
        messageQueue.put(message)
        print('learning done')

    def halt_learning(self):
        self.isHalted = True

    class Message:
        def __init__(self, image, done=False):
            self.image = image
            self.done = done