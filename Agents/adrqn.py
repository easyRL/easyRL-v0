import numpy as np

from Agents import drqn
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import ActionTransitionFrame

class ADRQN(drqn.DRQN):
    displayName = 'ADRQN'

    def __init__(self, *args):
        super().__init__(*args)
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, ActionTransitionFrame(-1, empty_state, -1, 0, empty_state, False), history_length = self.historylength)

    def getRecentAction(self):
        return self.memory.get_recent_action()

    def choose_action(self, state):
        recent_state = self.getRecentState()
        recent_state = np.concatenate([recent_state[1:], [state]], 0)
        recentRawActions = self.getRecentAction()
        recent_action = [self.create_one_hot(self.action_size, action) for action in recentRawActions]
        qval = self.predict((recent_state, recent_action), False)
        action = np.argmax(qval)
        return action

    def addToMemory(self, state, action, reward, new_state, done):
        prev_action = self.memory.peak_frame().action
        self.memory.append_frame(ActionTransitionFrame(prev_action, state, action, reward, new_state, done))

    def predict(self, state, isTarget):
        state, action = state
        stateShape = (1,) + (self.historylength,) + self.state_size
        actionShape = (1,) + (self.historylength,) + (self.action_size,)
        state = np.reshape(state, stateShape)
        action = np.reshape(action, actionShape)

        if isTarget:
            result = self.target.predict([state, action, self.allMask])
        else:
            result = self.model.predict([state, action, self.allMask])
        return result

    def buildQNetwork(self):
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Conv2D
        from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, concatenate, multiply

        input_shape = (self.historylength,) + self.state_size
        inputA = Input(shape=input_shape)
        inputB = Input(shape=(self.historylength, self.action_size))
        inputC = Input(shape=(self.action_size,))

        if len(self.state_size) == 1:
            x = TimeDistributed(Dense(24, activation='relu'))(inputA)
        else:
            x = TimeDistributed(Conv2D(16, 8, strides=4, activation='relu'))(inputA)
            x = TimeDistributed(Conv2D(32, 4, strides=2, activation='relu'))(x)

        x = TimeDistributed(Flatten())(x)
        x = Model(inputs=inputA, outputs=x)

        y = TimeDistributed(Dense(24, activation='relu'))(inputB)
        y = Model(inputs=inputB, outputs=y)

        combined = concatenate([x.output, y.output])

        z = LSTM(256)(combined)
        z = Dense(10, activation='relu')(z)  # fully connected
        z = Dense(10, activation='relu')(z)
        z = Dense(self.action_size)(z)
        outputs = multiply([z, inputC])

        inputs = [inputA, inputB, inputC]
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001, clipvalue=1))
        return model

    def calculateTargetValues(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + (self.historylength,) + self.state_size),
                   np.zeros((self.batch_size,) + (self.historylength,) + (self.action_size,)),
                   np.zeros((self.batch_size,) + (self.action_size,))]
        next_states = [np.zeros((self.batch_size,) + (self.historylength,) + self.state_size),
                   np.zeros((self.batch_size,) + (self.historylength,) + (self.action_size,))]

        for index_rep, history in enumerate(mini_batch):
            for histInd, transition in enumerate(history):
                X_train[0][index_rep][histInd] = transition.state
                next_states[0][index_rep][histInd] = transition.next_state
                X_train[1][index_rep][histInd] = self.create_one_hot(self.action_size, transition.prev_action)
                next_states[1][index_rep][histInd] = self.create_one_hot(self.action_size, transition.action)
            X_train[2][index_rep] = self.create_one_hot(self.action_size, transition.action)

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        qnext = self.target.predict(next_states + [self.allBatchMask])
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
        states = [np.zeros((self.batch_size,) + (self.historylength,) + self.state_size),
                  np.zeros((self.batch_size,) + (self.historylength,) + (self.action_size,))]
        for batch_idx, history in enumerate(mini_batch):
            for hist_idx, transition in enumerate(history):
                states[0][batch_idx][hist_idx] = transition.state
                states[1][batch_idx][hist_idx] = self.create_one_hot(self.action_size, transition.action)
        # Get the actions from the batch.
        actions = [history[-1].action for history in mini_batch]
        
        '''
        If the q_target is None then calculate the target q-value using the
        target QNetwork.
        '''
        if (q_target is None):
            next_states = [np.zeros((self.batch_size,) + (self.historylength,) + self.state_size),
                           np.zeros((self.batch_size,) + (self.historylength,) + (self.action_size,))]
            for batch_idx, history in enumerate(mini_batch):
                for hist_idx, transition in enumerate(history):
                    next_states[0][batch_idx][hist_idx] = transition.next_state
                    next_states[1][batch_idx][hist_idx] = self.create_one_hot(self.action_size, transition.action)
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

class ADRQNPrioritized(ADRQN):
    displayName = 'ADRQN Prioritized'
    newParameters = [ADRQN.Parameter('Alpha', 0.00, 1.00, 0.001, 0.60, True, True, "The amount of prioritization that gets used.")]
    parameters = ADRQN.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(ADRQNPrioritized.newParameters)
        super().__init__(*args[:-paramLen])
        self.alpha = float(args[-paramLen])
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.PrioritizedReplayBuffer(self, self.memory_size, ActionTransitionFrame(-1, empty_state, -1, 0, empty_state, False),
                                                                history_length = self.historylength, alpha = self.alpha)
        
class ADRQNHindsight(ADRQN):
    displayName = 'ADRQN Hindsight'
    newParameters = []
    parameters = ADRQN.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(ADRQNHindsight.newParameters)
        super().__init__(*args)
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.HindsightReplayBuffer(self, self.memory_size, ActionTransitionFrame(-1, empty_state, -1, 0, empty_state, False),
                                                                history_length = self.historylength)
                                                              
       
