from Agents import deepQ
import numpy as np
import random
from collections import deque
import itertools


class DRQN(deepQ.DeepQ):
    displayName = 'DRQN'
    newParameters = [deepQ.DeepQ.Parameter('History Length', 0, 20, 1, 10, True, True, "The number of recent timesteps to use as input")]
    parameters = deepQ.DeepQ.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DRQN.newParameters)
        self.historylength = int(args[-paramLen])
        super().__init__(*args[:-paramLen])
        self.memory = DRQN.ReplayBuffer(self, self.memory_size, self.historylength)

    def getRecentState(self):
        return self.memory.get_recent_state()

    def resetBuffer(self):
        self.memory = DRQN.ReplayBuffer(self, self.memory_size, self.historylength)

    def buildQNetwork(self):
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Conv2D
        from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply

        input_shape = (self.historylength,) + self.state_size
        inputA = Input(shape=input_shape)
        inputB = Input(shape=(self.action_size,))

        if len(self.state_size) == 1:
            x = TimeDistributed(Dense(10, input_shape=input_shape, activation='relu'))(inputA)
        else:
            x = TimeDistributed(Conv2D(16, 8, strides=4, activation='relu'))(inputA)
            x = TimeDistributed(Conv2D(32, 4, strides=2, activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(256)(x)
        x = Dense(10, activation='relu')(x)  # fully connected
        x = Dense(10, activation='relu')(x)
        x = Dense(self.action_size)(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001, clipvalue=1))
        return model

    def calculateTargetValues(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + (self.historylength,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        next_states = np.zeros((self.batch_size,) + (self.historylength,) + self.state_size)

        for index_rep, history in enumerate(mini_batch):
            for histInd, (state, action, reward, next_state, isDone) in enumerate(history):
                X_train[0][index_rep][histInd] = state
                next_states[index_rep][histInd] = np.array(next_state)
            X_train[1][index_rep] = self.create_one_hot(self.action_size, action)

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        qnext = self.target.predict([next_states, self.allBatchMask])
        qnext = np.amax(qnext, 1)

        for index_rep, history in enumerate(mini_batch):
            state, action, reward, next_state, isDone = history[-1]
            if isDone:
                Y_train[index_rep][action] = reward
            else:
                Y_train[index_rep][action] = reward + qnext[index_rep] * self.gamma
        return X_train, Y_train

    def choose_action(self, state):
        state = np.array(state)
        recent_state = self.getRecentState()
        recent_state = np.concatenate([recent_state[1:], [state]], 0)
        return super().choose_action(recent_state)

    def predict(self, state, isTarget):
        import tensorflow as tf

        shape = (1,) + (self.historylength,) + self.state_size
        state = np.reshape(state, shape)
        state = tf.cast(state, dtype=tf.float32)
        if isTarget:
            result = self.target.predict([state, self.allMask])
        else:
            result = self.model.predict([state, self.allMask])
        return result

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.appendFrame(state, action, reward, new_state, done)


    class ReplayBuffer:
        def __init__(self, learner, maxlength, historylength):
            self.learner = learner
            self.maxlength = maxlength
            self.historylength = historylength
            self.transitions = [None]*self.maxlength
            self.curIndex = 0
            emptyState = self.getEmptyState()
            self.emptyTrans = (emptyState, -1, 0, emptyState, False)

        def __len__(self):
            return self.curIndex

        def getEmptyState(self):
            shape = self.learner.state_size
            if len(shape) >= 2:
                return [[[-10000]] * shape[0] for _ in range(shape[1])]
            return [-10000] * shape[0]

        def appendFrame(self, state, action, reward, next_state, isdone):
            self.transitions[self.curIndex % self.maxlength] = (state, action, reward, next_state, isdone)
            self.curIndex += 1

        def getTransitions(self, startInd):
            result = []
            limit = (self.curIndex-1)%self.maxlength
            for ind in range(startInd, startInd+self.historylength):
                ind %= self.maxlength
                transition = self.transitions[ind]
                if transition is None:
                    break
                result.append(transition)
                if transition[4] or ind == limit:
                    break
            return self.pad(result)

        def pad(self, transitions):
            pad = [self.emptyTrans for _ in
                   range(self.historylength - len(transitions))]
            return pad + transitions

        def sample(self, batch_size):
            result = []
            for ind in random.sample(range(min(self.maxlength, self.curIndex)), batch_size):
                result.append(self.getTransitions(ind))
            return result

        def get_recent_state(self):
            result = [None]*self.historylength
            resInd = self.historylength-1
            start = (self.curIndex-1)%self.maxlength
            transition = self.transitions[start]
            result[resInd] = transition[0]
            resInd -= 1
            for ind in range(start-1, start-self.historylength, -1):
                ind %= self.maxlength
                transition = self.transitions[ind]
                if not transition or transition[4]:
                    break
                result[resInd] = transition[0]
                resInd -= 1

            emptyState = self.getEmptyState()

            while resInd >= 0:
                result[resInd] = emptyState
                resInd -= 1

            return result