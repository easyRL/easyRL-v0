from Agents import drqn
import numpy as np
import itertools

class ADRQN(drqn.DRQN):
    displayName = 'ADRQN'

    def __init__(self, *args):
        super().__init__(*args)
        self.memory = ADRQN.ReplayBuffer(self, 4000, self.historylength)
        self.target_update_interval = 1000

    def getRecentAction(self):
        return self.memory.get_recent_action()

    def choose_action(self, state):
        recent_state = self.getRecentState()
        recent_state = np.concatenate([[state], recent_state[:-1]], 0)
        recentRawActions = self.getRecentAction()
        recent_action = [self.create_one_hot(self.action_size, action) for action in recentRawActions]
        qval = self.predict((recent_state, recent_action), False)
        action = np.argmax(qval)
        return action

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros((vector_length))
        if hot_index != -1:
            output[hot_index] = 1
        return output

    def predict(self, state, isTarget):
        state, action = state
        stateShape = (1,) + (self.historylength,) + self.state_size
        actionShape = (1,) + (self.historylength,) + (self.action_size,)
        state = np.reshape(state, stateShape)
        action = np.reshape(action, actionShape)

        if isTarget:
            result = self.target.predict([state, action])
        else:
            result = self.model.predict([state, action])
        return result

    def buildQNetwork(self):

        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Conv2D
        from tensorflow.keras.layers import MaxPool2D, Flatten, TimeDistributed, LSTM, concatenate

        input_shape = (self.historylength,) + self.state_size
        inputA = Input(shape=input_shape)
        inputB = Input(shape=(self.historylength, self.action_size))
        # x = TimeDistributed(Dense(24, activation='relu'))(inputA)
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputA)
        x = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
        x = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(Flatten())(x)
        x = Model(inputs=inputA, outputs=x)

        y = TimeDistributed(Dense(10, activation='relu'))(inputB)
        y = Model(inputs=inputB, outputs=y)

        combined = concatenate([x.output, y.output])

        z = LSTM(512)(combined)
        z = Dense(24, activation='relu')(z)  # fully connected
        z = Dense(24, activation='relu')(z)
        outputs = Dense(self.action_size)(z)

        inputs = [inputA, inputB]
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001, clipvalue=1))
        return model, inputs, outputs

    def calculateTargetValues(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + (self.historylength,) + self.state_size),
                   np.zeros((self.batch_size,) + (self.historylength,) + (self.action_size,))]
        next_states = [np.zeros((self.batch_size,) + (self.historylength,) + self.state_size),
                   np.zeros((self.batch_size,) + (self.historylength,) + (self.action_size,))]

        for index_rep, history in enumerate(mini_batch):
            for histInd, (prevAction, state, action, reward, next_state, isDone) in enumerate(history):
                X_train[0][index_rep][histInd] = state
                next_states[0][index_rep][histInd] = next_state
                X_train[1][index_rep][histInd] = self.create_one_hot(self.action_size, prevAction)
                next_states[1][index_rep][histInd] = self.create_one_hot(self.action_size, action)

        Y_train = self.model.predict(X_train)
        qnext = self.target.predict(next_states)

        for index_rep, history in enumerate(mini_batch):
            prevAction, state, action, reward, next_state, isDone = history[0]
            if isDone:
                Y_train[index_rep][action] = reward
            else:
                Y_train[index_rep][action] = reward + np.amax(qnext[index_rep]) * self.gamma
        return X_train, Y_train

    class ReplayBuffer(drqn.DRQN.ReplayBuffer):
        def getTransitions(self, episode, startInd):
            base = list(itertools.islice(episode, startInd, min(len(episode), startInd + self.historylength)))
            for i, state in enumerate(base):
                ind = startInd+i
                prevAction = episode[ind-1][1] if ind > 0 else -1
                base[i] = (prevAction,) + base[i]

            shape = self.learner.state_size
            emptyState = np.array([[[-10000]] * shape[0] for _ in range(shape[1])])
            # emptyState = np.array([-10000] * shape[0])
            pad = [[-1, emptyState, -1, 0, emptyState, False] for _ in
                   range(max(0, (startInd + self.historylength - len(episode))))]
            return base+pad

        def get_recent_action(self):
            episode = self.currentEpisodes[self.curEpisodeNumber%len(self.currentEpisodes)]
            result = self.getTransitions(episode, 0)
            result = [action for _, _, action, _, _, _ in result]
            return result

        def get_recent_state(self):
            episode = self.currentEpisodes[self.curEpisodeNumber%len(self.currentEpisodes)]
            result = self.getTransitions(episode, 0)
            result = [state for _, state, _, _, _, _ in result]
            return result