import numpy as np

from Agents import deepQ
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame


class DRQN(deepQ.DeepQ):
    displayName = 'DRQN'
    newParameters = [deepQ.DeepQ.Parameter('History Length', 0, 20, 1, 10, True, True, "The number of recent timesteps to use as input")]
    parameters = deepQ.DeepQ.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DRQN.newParameters)
        self.historylength = int(args[-paramLen])
        super().__init__(*args[:-paramLen])
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False), history_length = self.historylength)

    def getRecentState(self):
        return self.memory.get_recent_state()

    def resetBuffer(self):
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, self.historylength)

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
            for histInd, transition in enumerate(history):
                X_train[0][index_rep][histInd] = transition.state
                next_states[index_rep][histInd] = np.array(transition.next_state)
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        qnext = self.target.predict([next_states, self.allBatchMask])
        qnext = np.amax(qnext, 1)

        for index_rep, history in enumerate(mini_batch):
            transition = history[-1]
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma
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
