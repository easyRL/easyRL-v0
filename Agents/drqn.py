from Agents import deepQ
import numpy as np
import random
from collections import deque
import itertools


class DRQN(deepQ.DeepQ):
    displayName = 'DRQN'

    def __init__(self, *args):
        self.historylength = 10
        super().__init__(*args)
        self.batch_size = 32
        self.memory = DRQN.ReplayBuffer(self, 4000, self.historylength)

    def getRecentState(self):
        return self.memory.get_recent_state()

    def resetBuffer(self):
        self.memory = DRQN.ReplayBuffer(self, 40, self.historylength)

    def buildQNetwork(self):
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Conv2D
        from tensorflow.keras.layers import MaxPool2D, Flatten, TimeDistributed, LSTM

        input_shape = (self.historylength,) + self.state_size
        inputs = Input(shape=input_shape)
        # x = TimeDistributed(Dense(10, input_shape=input_shape, activation='relu'))(inputs)
        x = TimeDistributed(Conv2D(32, (3,3), activation='relu'))(inputs)
        x = TimeDistributed(MaxPool2D(pool_size=(2,2)))(x)
        x = TimeDistributed(Conv2D(64, (3,3), activation='relu'))(x)
        x = TimeDistributed(MaxPool2D(pool_size=(2,2)))(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(512)(x)
        x = Dense(10, activation='relu')(x)  # fully connected
        x = Dense(10, activation='relu')(x)
        outputs = Dense(self.action_size)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001, clipvalue=1))
        return model, inputs, outputs

    def calculateTargetValues(self, mini_batch):
        X_train = np.zeros((self.batch_size,) + (self.historylength,) + self.state_size)
        next_states = np.zeros((self.batch_size,) + (self.historylength,) + self.state_size)

        for index_rep, history in enumerate(mini_batch):
            for histInd, (state, action, reward, next_state, isDone) in enumerate(history):
                X_train[index_rep][histInd] = state
                next_states[index_rep][histInd] = next_state

        Y_train = self.model.predict(X_train)
        qnext = self.target.predict(next_states)

        for index_rep, history in enumerate(mini_batch):
            state, action, reward, next_state, isDone = history[0]
            if isDone:
                Y_train[index_rep][action] = reward
            else:
                Y_train[index_rep][action] = reward + np.amax(qnext[index_rep]) * self.gamma
        return X_train, Y_train

    def choose_action(self, state):
        recent_state = self.getRecentState()
        recent_state = np.concatenate([[state], recent_state[:-1]], 0)
        return super().choose_action(recent_state)

    def predict(self, state, isTarget):
        import tensorflow as tf

        shape = (1,) + (self.historylength,) + self.state_size
        state = np.reshape(state, shape)
        state = tf.cast(state, dtype=tf.float32)
        if isTarget:
            result = self.target.predict(state)
        else:
            result = self.model.predict(state)
        return result

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, episode, done):
        self.memory.appendFrame(state, action, reward, new_state, done, episode)


    class ReplayBuffer:
        def __init__(self, learner, maxlength, historylength):
            self.learner = learner
            self.maxlength = maxlength
            self.historylength = historylength
            self.currentEpisodes = [deque() for _ in range(self.maxlength)]
            self.curEpisodeNumber = 0
            self.totalentries = 0

        def __len__(self):
            return self.totalentries

        def appendFrame(self, state, action, reward, next_state, isdone, episodeNumber):
            curEpisode = self.currentEpisodes[episodeNumber % self.maxlength]
            curEpisode.appendleft((state, action, reward, next_state, isdone))
            self.totalentries += 1
            if isdone:
                curEpisode = self.currentEpisodes[(episodeNumber+1) % self.maxlength]
                self.totalentries -= len(curEpisode)
                curEpisode.clear()
                self.curEpisodeNumber += 1

        def getTransitions(self, episode, startInd):
            base = list(itertools.islice(episode, startInd, min(len(episode), startInd + self.historylength)))
            shape = self.learner.state_size
            # emptyState = np.array([[[-10000]] * shape[0] for _ in range(shape[1])])
            emptyState = np.array([-10000] * shape[0])
            pad = [(emptyState, -1, 0, emptyState, False) for _ in
                   range(max(0, (startInd + self.historylength - len(episode))))]
            return base+pad

        def sample(self, batch_size):
            filledEpisodes = [episode for episode in self.currentEpisodes[:min(self.curEpisodeNumber+1,len(self.currentEpisodes))] if episode]
            episodes = random.choices(filledEpisodes, k=batch_size)
            result = []
            for episode in episodes:
                startInd = random.randrange(len(episode))
                result.append(self.getTransitions(episode, startInd))
            return result

        def get_recent_state(self):
            episode = self.currentEpisodes[self.curEpisodeNumber%len(self.currentEpisodes)]
            result = self.getTransitions(episode, 0)
            result = [state for state, _, _, _, _ in result]
            return result