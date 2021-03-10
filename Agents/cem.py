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
                     policyIteration.PolicyIteration.Parameter('Population Size', 0, 100, 10, 10, True, True, "The size of the sample population"),
                     policyIteration.PolicyIteration.Parameter('Elite Fraction', 0.001, 1.0, 0.001, 0.2, True, True, "The proportion of the elite to consider for policy improvement.")]
    parameters = policyIteration.PolicyIteration.parameters + newParameters


    def __init__(self,*args):
        paramLen = len(CEM.newParameters)
        super().__init__(*args[:-paramLen])
        self.sigma, self.pop_size, self.elite_frac = args[-paramLen:]
        # Convert the pop_size to an integer.
        self.pop_size = int(self.pop_size) 
        # Calculate the number of weight sets to consider the elite.
        self.elite = int(self.pop_size*self.elite_frac)
        
        '''
        Define network 
        '''
        # Hidden layer size
        self.hidden_size = 16
        # define layers
        self.layer1 = nn.Linear(self.state_size[0], self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.action_size)
        # Weights of the model
        self.best_weights = self.sigma*np.random.randn(self._get_weights_dim())
        self.weights_samples = [self.best_weights + (self.sigma*np.random.randn(self._get_weights_dim())) for i in range(self.pop_size)]
        self.set_policy(self.best_weights)
        
    def choose_action(self, state):
        # Convert the state to a tensor.
        state = torch.from_numpy(state).float()
        # Predict the value of each action using the network.
        x = F.relu(self.layer1(state))
        x = F.tanh(self.layer2(x))
        # Choose the action with the maximum predicted value.
        action = np.argmax(np.array(x.cpu().data))
        # Return the chosen action.
        return action

    def update(self, rewards):
        if (len(rewards) != self.pop_size):
            raise ValueError("The length of the list of rewards should be equal to the population size.")
        # Update the best weights based on the give rewards.
        elite_idxs = rewards.argsort()[-self.elite:]
        elite_weights = [self.weights_samples[i] for i in elite_idxs]
        self.best_weights = np.array(elite_weights).mean(axis=0)
        self.weights_samples = [self.best_weights + (self.sigma*np.random.randn(self._get_weights_dim())) for i in range(self.pop_size)]
        self.set_policy(self.best_weights)

    def get_policies(self):
        return self.weights_samples
        
    def set_policy(self, weights):
        # Separate the weights for each layer.
        layer1_end = (self.state_size[0]*self.hidden_size)+self.hidden_size
        layer1_W = torch.from_numpy(weights[:self.state_size[0]*self.hidden_size].reshape(self.state_size[0], self.hidden_size))
        layer1_b = torch.from_numpy(weights[self.state_size[0]*self.hidden_size:layer1_end])
        layer2_W = torch.from_numpy(weights[layer1_end:layer1_end+(self.hidden_size*self.action_size)].reshape(self.hidden_size, self.action_size))
        layer2_b = torch.from_numpy(weights[layer1_end+(self.hidden_size*self.action_size):])
        # set the weights for each layer
        self.layer1.weight.data.copy_(layer1_W.view_as(self.layer1.weight.data))
        self.layer1.bias.data.copy_(layer1_b.view_as(self.layer1.bias.data))
        self.layer2.weight.data.copy_(layer2_W.view_as(self.layer2.weight.data))
        self.layer2.bias.data.copy_(layer2_b.view_as(self.layer2.bias.data))
    
    def _get_weights_dim(self):
        return (self.state_size[0] + 1) * self.hidden_size + (self.hidden_size + 1) * self.action_size

    def save(self, filename):
        pass

    def load(self, filename):
        pass
    
    def memsave(self):
        pass

    def memload(self, mem):
        pass
