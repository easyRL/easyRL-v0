from Agents import agent, modelFreeAgent
from Agents.deepQ import DeepQ
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply
from tensorflow.keras import utils
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam

class ppo:
    def __init__(self, parameters, newParameters, action_size, state_size, mini_batch, gamma, horizon, epoch, episodes, policy_lr, value_lr):
        self.newParameters = newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Policy learning rate', 0.00001, 1, 0.00001, 0.001, True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Value learning rate', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Horizon', 10, 10000, 1, 50,
                                                             True, True,
                                                             "The number of timesteps over which the returns are calculated"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Epoch Size', 10, 100000, 1, 500,
                                                             True, True,
                                                             "The length of each epoch (likely should be the same as the max episode length)"),
                     modelFreeAgent.ModelFreeAgent.Parameter('PPO Epsilon', 0.00001, 0.5, 0.00001, 0.2,
                                                             True, True,
                                                             "A measure of how much a policy can change w.r.t. the states it's trained on"),
                     modelFreeAgent.ModelFreeAgent.Parameter('PPO Lambda', 0.5, 1, 0.001, 0.95,
                                                             True, True,
                                                             "A parameter that when set below 1, can decrease variance while maintaining reasonable bias")]
        self.parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters
        self.action_size = agent.action_size
        self.state_size = agent.state_size
        self.mini_batch = mini_batch
        self.gamma = gamma
        self.horizon = horizon
        self.epoch = epoch
        self.episodes = episodes
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.rewards = []

    def buildQNetwork(self):
        inputA = Input(shape=self.state_size)
        inputB = Input(shape=(self.action_size,))

        if len(self.state_size) == 1:
            x = TimeDistributed(Dense(10, input_shape=self.state_size, activation='relu'))(inputA)
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
        #model.compile(loss=kl, optimizer=Adam(lr=0.0001, clipvalue=1))
        return model

    def calculateTargetValues(self, mini_batch):
        DeepQ.calculateTargetValues(mini_batch)

    def update(self):
        DeepQ.update()

    def remember(self, state, action, reward, new_state, done): 
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass