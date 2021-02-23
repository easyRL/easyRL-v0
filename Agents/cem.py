import cProfile
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agents import policyIteration
from collections import deque

class CEM(policyIteration.PolicyIteration):
    displayName = 'CEM'
    newParameters = [policyIteration.PolicyIteration.Parameter('Sigma', 0.001, 1.0, 0.001, 0.5, True, True, "The standard deviation of additive noise"),
                     policyIteration.PolicyIteration.Parameter('Population Size', 0, 1000, 10, 50, True, True, "The size of the sample population"),
                     policyIteration.PolicyIteration.Parameter('Elite Fraction', 0.001, 1.0, 0.001, 0.2, True, True, "The proportion of the elite to consider for policy improvement.")]
    parameters = policyIteration.PolicyIteration.parameters + newParameters


    def __init__(self,*args):
        paramLen = len(CEM.newParameters)
        super().__init__(*args[:-paramLen])
        self.sigma, self.pop_size, self.elite_frac = args[-paramLen:]       
        self.elite = int(self.pop_size*self.elite_frac)
        # Convert the pop_size to an integer.
        self.pop_size = int(self.pop_size) 
        
        '''
        Define network 
        '''
        # Hidden layer size
        self.h_size = 16
        # define layers
        self.fc1 = nn.Linear(self.state_size[0], self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.action_size)
        # Weights of the model
        self.best_weight = self.sigma*np.random.randn(self._get_weights_dim())
        self.weights_pop = [self.best_weight + (self.sigma*np.random.randn(self._get_weights_dim())) for i in range(self.pop_size)]
        self.set_policy(self.best_weight)
        
    def choose_action(self, state):
        # Convert the state to a tensor.
        state = torch.from_numpy(state).float()
        # Predict the value of each action using the network.
        x = F.relu(self.fc1(state))
        x = F.tanh(self.fc2(x))
        # Choose the action with the maximum predicted value.
        action = np.argmax(np.array(x.cpu().data))
        # Return the chosen action.
        return action

    def update(self, rewards):
        elite_idxs = rewards.argsort()[-self.elite:]
        elite_weights = [self.weights_pop[i] for i in elite_idxs]
        self.best_weight = np.array(elite_weights).mean(axis=0)
        self.weights_pop = [self.best_weight + (self.sigma*np.random.randn(self._get_weights_dim())) for i in range(self.pop_size)]
        self.set_policy(self.best_weight)

    def get_policies(self):
        return self.weights_pop
        
    def set_policy(self, weights):
        # Separate the weights for each layer.
        fc1_end = (self.state_size[0]*self.h_size)+self.h_size
        fc1_W = torch.from_numpy(weights[:self.state_size[0]*self.h_size].reshape(self.state_size[0], self.h_size))
        fc1_b = torch.from_numpy(weights[self.state_size[0]*self.h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(self.h_size*self.action_size)].reshape(self.h_size, self.action_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(self.h_size*self.action_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def _get_weights_dim(self):
        return (self.state_size[0]+1)*self.h_size + (self.h_size+1)*self.action_size

    def save(self, filename):
        pass

    def load(self, filename):
        pass
    
    def memsave(self):
        pass

    def memload(self, mem):
        pass

'''
Method for converting PyTorch NN to Tensorflow NN.
    def _build_network(self):
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
'''
