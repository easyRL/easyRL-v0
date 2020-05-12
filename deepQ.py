from tensorflow.python.keras.optimizer_v2.adam import Adam

import modelFreeAgent
import numpy as np
from collections import deque
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import utils

class DeepQ(modelFreeAgent.ModelFreeAgent):
    def __init__(self, input_size, output_size, learning_rate, gamma):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = 16
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = self.buildQNetwork()
        self.target = self.buildQNetwork()
        self.memory = deque(maxlen=2000)
        self.total_steps = 0
        self.target_update_interval = 20

    def choose_action(self, state):
        qval = self.model.predict(np.reshape(state,(1, self.input_size)))
        action = np.argmax(qval)
        return action

    def remember(self, state, action, reward, new_state, done=False):
        self.memory.append((state, action, reward, new_state, done))
        X_train = np.zeros((self.batch_size, self.input_size))
        Y_train = np.zeros((self.batch_size, self.output_size))
        loss = 0
        if len(self.memory) < self.batch_size:
            #print("memory insufficient for training")
            return loss
        mini_batch = random.sample(self.memory, self.batch_size)

        for index_rep in range(self.batch_size):
            state, action, reward, next_state, isDone = mini_batch[index_rep]
            X_train[index_rep] = state
            Y_train[index_rep] = self.calculateTargetValue(reward, next_state, isDone)
        loss = self.model.train_on_batch(X_train, Y_train)
        self.updateTarget()
        return loss*400

    def updateTarget(self):
        if self.total_steps >= self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.target.set_weights(self.model.get_weights())
            #print("target updated")

    def update(self):
        pass

    def reset(self):
        pass

    def buildQNetwork(self):
        model = Sequential()
        model.add(Dense(10, input_dim=self.input_size, activation='relu'))  # fully connected
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.output_size))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def calculateTargetValue(self, reward, next_state, isDone):
        qnext = self.target.predict(np.reshape(next_state, (1, self.input_size)))
        maxqval = max(qnext)

        if isDone:
            target_value = reward
        else:
            target_value = reward + self.gamma * maxqval

        return target_value

    def __deepcopy__(self, memodict={}):
        pass


