import joblib
import numpy as np
import random
import math

from Agents import modelFreeAgent
from Agents.deepQ import DeepQ
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

class DuelingDDQN(DeepQ):
    displayName = 'Double Dueling DeepQ'
    newParameters = [DeepQ.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     DeepQ.Parameter('Memory Size', 1, 655360, 1, 1000, True, True, "The maximum number of timestep transitions to keep stored"),
                     DeepQ.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True, "The distance in timesteps between target model updates"),
                     DeepQ.Parameter('Learning Rate', 0.00001, 100, 0.00001, 0.001, True, True, "The rate at which the parameters respond to environment observations")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args): 
        paramLen = len(duelingDDQN.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, self.memory_size, self.target_update_interval, _ = [int(arg) for arg in args[-paramLen:]]
        _, _, _, self.learning_rate = [arg for arg in args[-paramLen:]]
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)

        self.model = self.buildQNetwork()
        self.target = self.buildQNetwork()
        self.lr = 0.001

        # Parameters used for Bellman Distribution
        self.distribution_list = []
        self.num_atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms -1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def sample_trajectories(self):
        mini_batch = self.sample()
        states, actions, rewards, next_states, dones = mini_batch
        return states, actions, rewards, next_states, dones

    def choose_action(self, state):
        z = self.model.predict(state)
        z_concat = np.vstack(z)
        q_value = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        action = np.argmax(q_value)
        return action
        
    def compute_loss(self):
        states, actions, rewards, next_states, dones = self.sample_trajectories()
        best_actions = [] 
        probs = [np.zeros((self.batch_size, self.num_atoms)) for i in range(self.action_size)]

        z = self.model.predict(next_states)
        z_ = self.target.predict(next_states)
        best_actions = []
        z_concat = np.vstack(z)
        q_value = np.sum(np.multiply(z_concat, np.array(self.z)))
        q_value = q_value.reshape((self.batch_size, self.action_size), order='F')
        best_actions = np.argmax(q_value, axis=1)

        for i in range(self.batch_size):
            if dones[i]:
                # Compute target distribution
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z 
                m_l, m_u = math.floor(bj), math.ceil(bj)
                probs[actions[i]][i][int(m_l)] += (m_u - bj)
                probs[actions[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z 
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    probs[actions[i]][i][int(m_l)] += z_[best_actions[i]][i][j] * (m_u - bj)
                    probs[actions[i]][i][int(m_u)] += z_[best_actions[i]][i][j] * (bj - m_l)

        # Computes KL loss from predicted and target distribution
        loss = self.model.fit(states, probs, batch_size = self.batch_size, epochs = 1, verbose = 0)
        return loss

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

        inputA = Input(shape=self.state_size)
        inputB = Input(shape=(self.action_size,))
        x = Flatten()(inputA)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='softmax')(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        kl = tf.keras.losses.KLDivergence()
        model.compile(loss=kl, optimizer=Adam(lr=self.lr))
        self.distribution_list = []
        for i in range(self.action_size):
            self.distribution_list.append(Dense(self.num_atoms, activation='softmax')(x))
        return model

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))
    
    def predict(self, state, isTarget):
        pass

    def remember(self, state, action, reward, new_state, done):
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass