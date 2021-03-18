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
    
    def compute_loss(self, mini_batch, q_target: list = None):
        """
        Computes the loss of each sample in the mini_batch. The loss is
        calculated as the TD Error of the Q-Network Will use the given
        list of q_target value if provided instead of calculating.
        :param mini_batch: is the mini batch to compute the loss of.
        :param q_target: is a list of q_target values to use in the
        calculation of the loss. This is optional. The q_target values
        will be calculated if q_target is not provided.
        :type q_target: list
        """
        # Get the states from the batch.
        states = np.zeros((self.batch_size,) + (self.historylength,) + self.state_size)
        for batch_idx, history in enumerate(mini_batch):
            for hist_idx, transition in enumerate(history):
                states[batch_idx][hist_idx] = np.array(transition.state)
        # Get the actions from the batch.
        actions = [history[-1].action for history in mini_batch]
        
        '''
        If the q_target is None then calculate the target q-value using the
        target QNetwork.
        '''
        if (q_target is None):
            next_states = np.zeros((self.batch_size,) + (self.historylength,) + self.state_size)
            for batch_idx, history in enumerate(mini_batch):
                for hist_idx, transition in enumerate(history):
                    next_states[batch_idx][hist_idx] = np.array(transition.next_state)
            rewards = [history[-1].reward for history in mini_batch]
            is_dones = np.array([history[-1].is_done for history in mini_batch]).astype(float)
            q_target = self.target.predict([next_states, self.allBatchMask])
            q_target = rewards + (1 - is_dones) * self.gamma * np.amax(q_target, 1)
        
        # Get from the current q-values from the QNetwork.
        q = self.model.predict([states, self.allBatchMask])
        q = np.choose(actions, q.T)
        
        # Calculate and return the loss (TD Error).
        loss = (q_target - q) ** 2
        return loss
    
    def apply_hindsight(self):
        '''
       The hindsight replay buffer method checks for 
       the instance, if instance found add to the memory
               '''
        if (isinstance(self.memory, ExperienceReplay.HindsightReplayBuffer)):
            self.memory.apply_hindsight()

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

class DRQNPrioritized(DRQN):
    displayName = 'DRQN Prioritized'
    newParameters = [DRQN.Parameter('Alpha', 0.00, 1.00, 0.001, 0.60, True, True, "The amount of prioritization that gets used.")]
    parameters = DRQN.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DRQNPrioritized.newParameters)
        super().__init__(*args[:-paramLen])
        self.alpha = float(args[-paramLen])
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.PrioritizedReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False),
                                                                history_length = self.historylength, alpha = self.alpha)
        
class DRQNHindsight(DRQN):
    displayName = 'DRQN Hindsight'
    newParameters = []
    parameters = DRQN.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DRQNHindsight.newParameters)
        super().__init__(*args)
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.HindsightReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False),
                                                                history_length = self.historylength)  
    
    
        
