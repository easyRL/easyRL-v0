import joblib
import numpy as np
import random

from Agents import modelFreeAgent
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

class DeepQ(modelFreeAgent.ModelFreeAgent):
    displayName = 'Deep Q'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True, "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True, "The distance in timesteps between target model updates")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DeepQ.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, self.memory_size, self.target_update_interval = [int(arg) for arg in args[-paramLen:]]
        self.model = self.buildQNetwork()
        self.target = self.buildQNetwork()
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)

    def choose_action(self, state):
        qval = self.predict(state, False)
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.time_steps)
        # TODO: Put epsilon at a level near this
        # if random.random() > epsilon:
        action = np.argmax(qval)
        # else:
            # action = self.state_size.sample()
        return action

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):
        self.addToMemory(state, action, reward, new_state, done)
        
        loss = 0
        if len(self.memory) < 2*self.batch_size:
            return loss
        batch_idxes, mini_batch = self.sample()

        X_train, Y_train = self.calculateTargetValues(mini_batch)
        loss = self.model.train_on_batch(X_train, Y_train)
        '''
        If the memory is PrioritiedReplayBuffer then calculate the loss and
        update the priority of the sampled transitions
        '''
        if (isinstance(self.memory, ExperienceReplay.PrioritizedReplayBuffer)):
            # Calculate the loss of the batch as the TD error
            td_errors = self.compute_loss(mini_batch, np.amax(Y_train, axis = 1))
            # Update the priorities.
            for idx, td_error in zip(batch_idxes, td_errors):
                self.memory.update_error(idx, td_error)
        self.updateTarget()
        return loss

    def updateTarget(self):
        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.target.set_weights(self.model.get_weights())
            print("target updated")
        self.total_steps += 1

    def predict(self, state, isTarget):

        shape = (1,) + self.state_size
        state = np.reshape(state, shape)
        if isTarget:
            result = self.target.predict([state, self.allMask])
        else:
            result = self.model.predict([state, self.allMask])
        return result

    def update(self):
        pass

    def reset(self):
        pass

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros((vector_length))
        if hot_index != -1:
            output[hot_index] = 1
        return output

    def buildQNetwork(self):
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, Flatten, multiply

        inputA = Input(shape=self.state_size)
        inputB = Input(shape=(self.action_size,))
        x = Flatten()(inputA)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def calculateTargetValues(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        next_states = np.zeros((self.batch_size,) + self.state_size)

        for index_rep, transition in enumerate(mini_batch):
            states, actions, rewards, _, dones = transition
            
            X_train[0][index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        qnext = self.target.predict([next_states, self.allBatchMask])
        qnext = np.amax(qnext, 1)

        for index_rep, transition in enumerate(mini_batch):
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma

        print("X train: " + str(X_train))
        print("Y train: " + str(Y_train))

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
        states = np.zeros((self.batch_size,) + self.state_size)
        for batch_idx, transition in enumerate(mini_batch):
            states[batch_idx] = transition.state
        # Get the actions from the batch.
        actions = [transition.action for transition in mini_batch]
        
        '''
        If the q_target is None then calculate the target q-value using the
        target QNetwork.
        '''
        if (q_target is None):
            next_states = np.zeros((self.batch_size,) + self.state_size)
            for batch_idx, transition in enumerate(mini_batch):
                next_states[batch_idx] = transition.next_state
            rewards = [transition.reward for transition in mini_batch]
            is_dones = np.array([transition.is_done for transition in mini_batch]).astype(float)
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

    def __deepcopy__(self, memodict={}):
        pass

    def save(self, filename):
        mem = self.model.get_weights()
        joblib.dump((DeepQ.displayName, mem), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != DeepQ.displayName:
            print('load failed')
        else:
            self.model.set_weights(mem)
            self.target.set_weights(mem)

    def memsave(self):
        return self.model.get_weights()

    def memload(self, mem):
        self.model.set_weights(mem)
        self.target.set_weights(mem)



class DeepQPrioritized(DeepQ):
    displayName = 'Deep Q Prioritized'
    newParameters = [DeepQ.Parameter('Alpha', 0.00, 1.00, 0.001, 0.60, True, True, "The amount of prioritization that gets used.")]
    parameters = DeepQ.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DeepQPrioritized.newParameters)
        super().__init__(*args[:-paramLen])
        self.alpha = float(args[-paramLen])
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.PrioritizedReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False), alpha = self.alpha)
        
        
class DeepQHindsight(DeepQ):
    displayName = 'Deep Q Hindsight'
    newParameters = []
    parameters = DeepQ.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DeepQHindsight.newParameters)
        super().__init__(*args)
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.HindsightReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
       
        
