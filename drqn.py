import deepQ
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Conv3D
from tensorflow.keras.layers import MaxPool1D, MaxPool2D, MaxPool3D, Flatten, TimeDistributed, LSTM, concatenate


class DRQN(deepQ.DeepQ):
    displayName = 'DRQN'

    def __init__(self, *args):
        self.historylength = 4
        super().__init__(*args)
        self.batch_size = 16
        self.memory = DRQN.ReplayBuffer(4000, self.historylength)

    def getRecentState(self):
        return self.memory.get_recent_state()

    def resetBuffer(self):
        self.memory = DRQN.ReplayBuffer(40, self.historylength)

    def buildQNetwork(self):
        input_shape = (self.historylength,) + self.state_size
        inputs = Input(shape=input_shape)
        # x = TimeDistributed(Dense(10, input_shape=input_shape, activation='relu'))(inputs)
        x = TimeDistributed(Conv2D(32, (3,3), input_shape=input_shape, activation='relu'))(inputs)
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
                if len(np.array(state).shape) == 1:
                    print('hello')
                X_train[index_rep][histInd] = state
                next_states[index_rep][histInd] = next_state

        Y_train = self.model.predict(X_train)
        qnext = self.target.predict(next_states)

        for index_rep, history in enumerate(mini_batch):
            state, action, reward, next_state, isDone = history[-1]
            if isDone:
                Y_train[index_rep][action] = reward
            else:
                Y_train[index_rep][action] = reward + np.amax(qnext[index_rep]) * self.gamma
        return X_train, Y_train

    def choose_action(self, state):
        recent_state = self.getRecentState()
        recent_state[3] = state
        return super().choose_action(recent_state)

    def predict(self, state, isTarget):
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

    # class ReplayBuffer:
    #     def __init__(self, maxlength, historylength):
    #         self.maxlength = maxlength
    #         self.historylength = historylength
    #         self.action = None
    #         self.observation = None
    #         self.reward = None
    #         self.done = None
    #         self.episodeNumber = None
    #         self.nextindex = -1
    #         self.totalentries = 0
    #         self.currentEpisodes = None  # dictionary containing Key=epsiode number, value = (startindex, number of transitions)
    #
    #     def __len__(self):
    #         return self.totalentries
    #
    #     def appendFrame(self, state, episodeNumber):
    #         if self.observation is None:
    #             self.action = np.empty([self.maxlength])
    #             self.observation = np.empty([self.maxlength] + list(state.shape), dtype=np.int32)
    #             self.reward = np.empty([self.maxlength])
    #             self.done = np.empty([self.maxlength])
    #             self.episodeNumber = np.negative(np.ones([self.maxlength]))
    #             self.currentEpisodes = {}
    #
    #         self.nextindex = (int)((self.nextindex + 1) % self.maxlength)
    #         if self.episodeNumber[
    #             self.nextindex] > -1:  # already some value is present in the buffer and you want to re-write it
    #
    #             episodeNumber_prev = self.episodeNumber[self.nextindex]
    #             (begin, end) = self.currentEpisodes[episodeNumber_prev]
    #             if end == 1:
    #                 self.currentEpisodes.pop(episodeNumber_prev)
    #             else:
    #                 end = end - 1
    #                 self.currentEpisodes[episodeNumber_prev] = [begin + 1, end]
    #
    #         self.observation[self.nextindex] = state
    #         self.episodeNumber[self.nextindex] = episodeNumber
    #         if episodeNumber in self.currentEpisodes:
    #             (begin, end) = self.currentEpisodes[episodeNumber]
    #             self.currentEpisodes[episodeNumber] = [begin, end + 1]
    #         else:
    #             self.currentEpisodes[episodeNumber] = [self.nextindex, 1]
    #
    #         self.totalentries = min(self.maxlength, self.totalentries + 1)
    #
    #     def appendEffect(self, action, reward, done):
    #         self.reward[self.nextindex] = reward
    #         self.action[self.nextindex] = action
    #         self.done[self.nextindex] = done
    #
    #     def getBatch(self, idxes):
    #         obs_batch = np.concatenate([self.concatFrames(idx)[np.newaxis, :] for idx in idxes], axis=0)
    #         next_obs_batch = np.concatenate([self.concatFrames(idx + 1)[np.newaxis, :] for idx in idxes], axis=0)
    #         result = [(obs_batch[ind], self.action[id], self.reward[id], next_obs_batch[ind], self.done[id]) for ind, id in enumerate(idxes)]
    #         return result
    #
    #     def sample(self, batch_size):
    #         # [print(self.currentEpisodes[x][1]) for x in self.currentEpisodes if self.currentEpisodes[x][1] >= batch_size]
    #         episodes = [x for x in self.currentEpisodes if self.currentEpisodes[x][1] >= batch_size]  # select the episode which contains atleast 2*batchsize transitions
    #         random_episode = random.sample(episodes, 1)[0]
    #
    #         (begin, end) = self.currentEpisodes[random_episode]
    #         idx = [x for x in range(begin, begin + end)]
    #         if (begin + end > self.maxlength):
    #             idx = [x for x in range(begin, self.maxlength)]
    #             idx1 = [x for x in range(0, begin + end - self.maxlength)]
    #             idx.extend(idx1)
    #
    #         if len(idx) < batch_size:
    #             idxes = random.choice(idx, batch_size)
    #         else:
    #             idxes = random.sample(idx, batch_size)
    #         return self.getBatch(idxes)
    #
    #     def get_recent_state(self):
    #         return self.concatFrames(int((self.nextindex - 1) % self.maxlength))
    #
    #     def concatFrames(self, idx):
    #         end_idx = idx  # + 1 # make noninclusive
    #         start_idx = end_idx - self.historylength
    #
    #         # insufficient frames in buffer
    #         if start_idx < 0 and self.totalentries != self.maxlength:
    #             start_idx = 0
    #         for idx in range(start_idx, end_idx - 1):
    #             if self.done[int(idx % self.maxlength)]:
    #                 start_idx = idx + 1
    #         # fill the missing history frames with zeros
    #         missing_context = self.historylength - (end_idx - start_idx)
    #         if start_idx < 0 or missing_context > 0:
    #             frames = [np.zeros_like(self.observation[0]) for _ in range(missing_context)]
    #             for idx in range(start_idx, end_idx):
    #                 frames.append(self.observation[int(idx % self.maxlength)])
    #             return np.reshape(frames, (self.historylength,) + self.observation.shape[1:])
    #         else:
    #             obsPart = self.observation[start_idx:end_idx]
    #             return obsPart.reshape((self.historylength,) + self.observation.shape[1:])

    class ReplayBuffer:
        def __init__(self, maxlength, historylength):
            self.maxlength = maxlength
            self.historylength = historylength
            self.currentEpisodes = [[] for _ in range(self.maxlength)]
            self.curEpisodeNumber = 0
            self.totalentries = 0

        def __len__(self):
            return self.totalentries

        def appendFrame(self, state, action, reward, next_state, isdone, episodeNumber):
            curEpisode = self.currentEpisodes[episodeNumber % self.maxlength]
            if episodeNumber == self.curEpisodeNumber-1:
                pass
            elif episodeNumber == self.curEpisodeNumber:
                self.totalentries -= len(curEpisode)
                curEpisode.clear()
                self.curEpisodeNumber += 1
            else:
                print('Error: should not occur')
            curEpisode.append([state, action, reward, next_state, isdone])
            self.totalentries += 1

        def getTransitions(self, episode, startInd):
            base = episode[startInd:min(len(episode), startInd + self.historylength)]
            shape = base[0][0].shape
            emptyState = np.array([[[-10000]] * shape[0] for _ in range(shape[1])])
            pad = [[emptyState, 0, 0, emptyState, base[-1][4]] for _ in
                   range(max(0, (startInd + self.historylength - len(episode))))]
            return base+pad

        def sample(self, batch_size):
            filledEpisodes = self.currentEpisodes[:min(self.curEpisodeNumber,len(self.currentEpisodes))]
            episodes = random.choices(filledEpisodes, k=batch_size)
            result = []
            for episode in episodes:
                startInd = random.randrange(len(episode))
                result.append(self.getTransitions(episode, startInd))
            return result

        def get_recent_state(self):
            episode = self.currentEpisodes[(self.curEpisodeNumber-1)%len(self.currentEpisodes)]
            result = self.getTransitions(episode, max(0, len(episode) - self.historylength+1))
            result = [state for state, _, _, _, _ in result]
            return result