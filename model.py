import random
import numpy as np


class Model:
    def __init__(self):
        # these can be set directly from the Controller based on user input from the View
        self.environment = None
        self.agent_class = None
        self.isHalted = False
        self.isRunning = False
        self.agent = None

    def run_learning(self, messageQueue, total_episodes, max_steps, gamma, learning_rate, max_epsilon, min_epsilon, decay_rate):
        self.isRunning = True
        epsilon = max_epsilon

        self.agent = self.agent_class(self.environment.state_size, self.environment.action_size, learning_rate, gamma)

        for episode in range(total_episodes):
            self.environment.reset()

            for step in range(max_steps):
                old_state = self.environment.state
                exp_exp_tradeoff = random.uniform(0, 1)

                if exp_exp_tradeoff > epsilon:
                    action = self.agent.choose_action(old_state)
                else:
                    action = self.environment.sample_action()

                reward = self.environment.step(action)

                loss = self.agent.remember(old_state, action, reward, self.environment.state)

                modelState = Model.State(self.environment.render(), epsilon, reward, loss)
                message = Model.Message(Model.Message.STATE, modelState)
                messageQueue.put(message)

                if self.environment.done or self.isHalted:
                    break

            message = Model.Message(Model.Message.EVENT, Model.Message.EPISODE)
            messageQueue.put(message)

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

            if self.isHalted:
                self.isHalted = False
                break
        message = Model.Message(Model.Message.EVENT, Model.Message.TRAIN_FINISHED)
        messageQueue.put(message)
        self.isRunning = False
        print('learning done')

    def run_testing(self, messageQueue, total_episodes, max_steps):
        self.isRunning = True

        if self.agent:
            for episode in range(total_episodes):
                self.environment.reset()

                for step in range(max_steps):
                    old_state = self.environment.state

                    action = self.agent.choose_action(old_state)

                    reward = self.environment.step(action)

                    modelState = Model.State(self.environment.render(), None, reward, None)
                    message = Model.Message(Model.Message.STATE, modelState)
                    messageQueue.put(message)

                    if self.environment.done or self.isHalted:
                        break

                message = Model.Message(Model.Message.EVENT, Model.Message.EPISODE)
                messageQueue.put(message)

                if self.isHalted:
                    self.isHalted = False
                    break
            message = Model.Message(Model.Message.EVENT, Model.Message.TEST_FINISHED)
            messageQueue.put(message)
            self.isRunning = False
            print('testing done')

    def halt_learning(self):
        if self.isRunning:
            self.isHalted = True

    class Message:
        # types of message
        STATE = 0
        EVENT = 1

        # event types
        TRAIN_FINISHED = 0
        TEST_FINISHED = 1
        EPISODE = 2

        def __init__(self, type, data):
            self.type = type
            self.data = data

    class State:
        def __init__(self, image, epsilon, reward, loss):
            self.image = image
            self.epsilon = epsilon
            self.reward = reward
            self.loss = loss