import joblib
import numpy as np
import random
import math

from Agents import modelFreeAgent
from Agents.deepQ import DeepQ
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame


# References:
# https://flyyufelix.github.io/2017/10/24/distributional-bellman.html
# http://github.com/flyyufelix/C51-DDQN-Keras/blob/master/c51_ddqn.py

class Rainbow(DeepQ):
    displayName = 'Rainbow'
    newParameters = [DeepQ.Parameter('Learning Rate', 0.00001, 100, 0.00001, 0.001, True, True, "The rate at which the parameters respond to environment observations")]
    parameters = DeepQ.parameters + newParameters

    def __init__(self, *args): 
        paramLen = len(Rainbow.newParameters)
        super().__init__(*args[:-paramLen])

        Qparams = []
        for i in range(3):
            Qparams.append(DeepQ.newParameters[i].default)
        '''self.batch_size, self.memory_size, self.target_update_interval = [int(param) for param in Qparams]
        #self.batch_size, self.memory_size, self.target_update_interval, _ = [int(arg) for arg in args[-paramLen:]]
        _, _, _, self.learning_rate = [arg for arg in args[-paramLen:]]
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)'''
        empty_state = self.get_empty_state()
        self.total_steps = 0
        self.model = self.buildQNetwork()
        self.target = self.buildQNetwork()
        self.lr = 0.001
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))

        # Parameters used for Bellman Distribution
        self.num_atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms -1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
        self.sample_size = min(self.batch_size, self.memory_size)
        # Initialize prioritization exponent
        self.p = 0.5
        self.allBatchMask = np.full((self.sample_size, self.num_atoms), 1)

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):
        self.addToMemory(state, action, reward, new_state, done)
        loss = 0
        if len(self.memory) < 2*self.batch_size:
            return loss
        batch_idx, mini_batch = self.sample()

        #X_train, Y_train = self.calculateTargetValues(mini_batch)
        #self.model.train_on_batch(X_train, Y_train)
        loss = self.agent_loss()
        if (isinstance(self.memory, ExperienceReplay.PrioritizedReplayBuffer)):
            errors = self.compute_loss(mini_batch)
            for idx, error in batch_idx, errors:
                self.memory.update_error(idx, error)
        return loss

    def sample_trajectories(self):
        _, mini_batch = self.sample()
        allStates = np.zeros(((self.sample_size, ) + self.state_size))
        allNextStates = np.zeros(((self.sample_size, ) + self.state_size))
        allActions = []
        allRewards = []
        allDones = []
        for index, transition in enumerate(mini_batch):
            states, actions, rewards, next_states, dones = transition
            allStates[index, :] = states
            allActions.append(actions)
            allRewards.append(rewards)
            allNextStates[index, :] = next_states
            allDones.append(dones)
        return allStates, allActions, allRewards, allNextStates, allDones

    def choose_action(self, state):
        shape = (1,) + self.state_size
        state = np.reshape(state, shape)
        z = self.model.predict([state])
        z_concat = np.vstack(z)
        q_value = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        action = np.argmax(q_value)
        return action
        
    def agent_loss(self):
        import tensorflow as tf

        allStates, allActions, allRewards, allNextStates, allDones = self.sample_trajectories()
        best_actions = [] 
        probs = [np.zeros((self.sample_size, self.num_atoms)) for i in range(self.action_size)]

        z = self.model.predict(allNextStates)
        z_ = self.target.predict(allNextStates)
        best_actions = []
        z_concat = np.vstack(z)
       # x = np.expand_dims(np.array(z_concat), axis=self.num_atoms)
       # x = np.expand_dims(np.array(self.z), axis=self.state_size[0])
        q_value = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q_value = q_value.reshape((self.sample_size, self.action_size), order='F')
        best_actions = np.argmax(q_value, axis=1)

        for i in range(self.sample_size):
            if allDones[i]:
                # Compute target distribution
                Tz = min(self.v_max, max(self.v_min, allRewards[i]))
                bj = (Tz - self.v_min) / self.delta_z 
                m_l, m_u = math.floor(bj), math.ceil(bj)
                probs[allActions[i]][i][int(m_l)] += (m_u - bj)
                probs[allActions[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, allRewards[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z 
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    probs[allActions[i]][i][int(m_l)] += z_[best_actions[i]][i][j] * (m_u - bj)
                    probs[allActions[i]][i][int(m_u)] += z_[best_actions[i]][i][j] * (bj - m_l)

        # Computes KL loss from predicted and target distribution
        loss = self.model.fit(allStates, probs, batch_size = self.sample_size, epochs = 1, verbose = 0)
        '''mini = 100000
        for loss in range(len(loss.history['loss'])):
            if mini > loss: 
                mini = loss '''
        losses = []
        for l in range(len(loss.history['loss'])):
            losses.append(l)
        #print(loss.history['loss'])
        return np.mean(np.array(losses))

    def updateTarget(self):
        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.target.set_weights(self.model.get_weights())
            print("target updated")
        self.total_steps += 1
    
    def buildQNetwork(self):
        import tensorflow as tf
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, Flatten, multiply

        inputs = Input(shape=self.state_size)
        #inputs = Flatten()(inputs)
        h1 = Dense(64, activation='relu')(inputs)
        h2 = Dense(64, activation='relu')(h1)
        outputs = []
        for _ in range(self.action_size):
            outputs.append(Dense(51, activation='softmax')(h2))
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=0.0000625, epsilon=1.5 * 1e-4))
        return model

    def calculateTargetValues(self, mini_batch):
        X_train = [np.zeros((self.sample_size,) + self.state_size), np.zeros((self.sample_size,) + (self.action_size,))]
        next_states = np.zeros((self.sample_size,) + self.state_size)

        for index_rep, transition in enumerate(mini_batch):
            states, actions, rewards, _, dones = transition
            
            X_train[0][index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state

        Y_train = np.zeros((self.sample_size,) + (self.action_size,))
        z = self.target.predict(next_states)
        z_concat = np.vstack(z)
        qnext = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        for index_rep, transition in enumerate(mini_batch):
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma
        #print("X train: " + str(X_train))
        #print("Y train: " + str(Y_train))
        return X_train, Y_train

    # compute td error for priority replay buffer
    def compute_loss(self, mini_batch):
        errors = []
        index = 0
        # sample transition at recent time step
        for time, sample in mini_batch:
            # reshapes the state and next_state
            state, next_state = sample.state, sample.next_state
            next_reward = self.memory.get_transitions(time+1).reward
            shape = (1,) + self.state_size
            state = np.reshape(state, shape)
            next_state = np.reshape(next_state, shape)

            # Retrieves necessary q values to compute error
            q = self.target.predict([state])
            q_next = self.model.predict([next_state])
            q_max = np.argmax(q_next)

            # Calculates td error which is used to compute priority
            error = (next_reward + self.gamma * q_max - q) ** self.p
            # self.memory.update_error(index, error)
            errors[index] = error
            index+=1
        return errors

    def save(self, filename):
        mem = self.model.get_weights()
        joblib.dump((Rainbow.displayName, mem), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != Rainbow.displayName:
            print('load failed')
        else:
            self.model.set_weights(mem)
            self.target.set_weights(mem)

    def memsave(self):
        return self.model.get_weights()

    def memload(self, mem):
        self.model.set_weights(mem)
        self.target.set_weights(mem)

    def predict(self, state, isTarget):
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass
