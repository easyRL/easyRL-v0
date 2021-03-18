from Agents import agent, modelFreeAgent
from Agents.deepQ import DeepQ
from Agents.models import Actor, Critic
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply
from tensorflow.keras import utils
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam
#from tensorflow_probability.distributions import Multinomial

class PPO(DeepQ):
    displayName = 'PPO'
    newParameters = [DeepQ.Parameter('Policy learning rate', 0.00001, 1, 0.00001, 0.001, True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     DeepQ.Parameter('Value learning rate', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     DeepQ.Parameter('Horizon', 10, 10000, 1, 50,
                                                             True, True,
                                                             "The number of timesteps over which the returns are calculated"),
                     DeepQ.Parameter('Epoch Size', 10, 100000, 1, 500,
                                                             True, True,
                                                             "The length of each epoch (likely should be the same as the max episode length)"),
                     DeepQ.Parameter('PPO Epsilon', 0.00001, 0.5, 0.00001, 0.2,
                                                             True, True,
                                                             "A measure of how much a policy can change w.r.t. the states it's trained on"),
                     DeepQ.Parameter('PPO Lambda', 0.5, 1, 0.001, 0.95,
                                                             True, True,
                                                            "A parameter that when set below 1, can decrease variance while maintaining reasonable bias")]
    parameters = DeepQ.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(PPO.newParameters)
        super().__init__(*args[:-paramLen])
        empty_state = self.get_empty_state()
        # Initialize parameters
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.total_steps = 0
        self.actorIts = 2
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)
        self.policy_lr = 0.001
        self.value_lr = 0.001
        self.policy_model = Actor(self.state_size, self.action_size, self.policy_lr).policy_network()
        self.value_model = Critic(self.state_size, self.action_size, self.value_lr).value_network()

    def sample(self):
        return self.memory.sample(self.actorIts)

    def policy_network(self):
        return self.policy_model

    def value_network(self):
        return self.value_model

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def choose_action(self, state):
        shape = (1,) + self.state_size
        state = np.reshape(state, shape)
        val = self.value_model.predict([state, self.allMask])
        action = np.argmax(val)
        print("action: " + str(action))
        return action

    def get_probabilities(self, states):
        probabilities = self.policy_model.predict([states, self.allBatchMask])
        return probabilities

    def remember(self, state, action, reward, new_state, done): 
        pass

    def predict(self, state, isTarget):
        pass

    def update(self):
        pass

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros((vector_length))
        if hot_index != -1:
            output[hot_index] = 1
        return output

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass
