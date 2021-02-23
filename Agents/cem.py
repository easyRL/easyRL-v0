import cProfile
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agents import policyIteration
from collections import deque

class CEM(nn.Module):
    displayName = 'Cross Entropy Method'
    newParameters = [policyIteration.PolicyIteration.Parameter('Sigma', 0, 1.0, 0.001, 0.5, True, True, "The standard deviation of additive noise"),
    policyIteration.PolicyIteration.Parameter('Population Size', 0, 1.0, 0.001, 0.5, True, True, "The standard deviation of additive noise"),
    policyIteration.PolicyIteration.Parameter('Elite Fraction', 0, 1.0, 0.001, 0.5, True, True, "The standard deviation of additive noise")]
    parameters =  policyIteration.PolicyIteration.parameters + newParameters


    def __init__(self,*args) 
        paramLen = len(CEM.newparameters)
        super()._init_(*args[:-paramLen])
        self.sigma, self.pop_size, self.elite_frac = [int(arg) for arg in args[-paramLen:]]
        
        self.elite = int(self.pop_size*self.elite_frac)
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = 16 #Hidden layer size
        self.a_size = env.action_space.shape[0]
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        # Weights of the model
        self.best_weight = self.sigma*np.random.randn(self._get_weights_dim())
        self.weights_pop = [self.best_weight + (self.sigma*np.random.randn(self._get_weights_dim())) for i in range(self.pop_size)]
        self.set_policy(self.best_weight)
        
    def choose_action(self, state, best=True):
        x = F.relu(self.fc1(state))
        x = F.tanh(self.fc2(x))
        return x.cpu().data

    def update(self, rewards):
        elite_idxs = rewards.argsort()[-self.elite:]
        elite_weights = [self.weights_pop[i] for i in elite_idxs]
        self.best_weight = np.array(elite_weights).mean(axis=0)
        self.weights_pop = [self.best_weight + (self.sigma*np.random.randn(self._get_weights_dim())) for i in range(self.pop_size)]
        self.set_policy(self.best_weight)

    def get_policies(self):
        return self.weights_pop
        
    def set_policy(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))


    def_build_network(self):
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, Flatten, multiply
        inputA = Input(shape=self.state_size)
        inputB = Input(shape=(self.action_size,))
        x = Flatten()(inputA)
        x = Dense(16, input_dim=self.state_size, activation='relu')(x)  # fully connected     
        x = Dense(self.action_size, activation='tanh')(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    
    
    def _get_weights_dim(self):
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
