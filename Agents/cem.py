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
    newParameters = [policyIteration.PolicyIteration.Parameter('Sigma', 0, 1.0, 0.001, 0.5, True, True, "The standard deviation of additive noise")]
    parameters =  policyIteration.PolicyIteration.parameter + newParameters

    def _init_(self, args):
        paramLen = len(CEM.newparameters)
        super()._init_(*args[:-paramLen])
        self.sigma = float(args[-paramLen])

    def choose_action(self, state):
        self.time_steps += 1
        return random.randrange(self.action_size)
        
        
        

  
def train(n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
    """PyTorch implementation of the cross-entropy method.
        
    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    best_weight = sigma*np.random.randn(agent.get_weights_dim())

    for i_iteration in range(1, n_iterations+1):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)
        
        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        if i_iteration % print_every == 0:
           print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
           print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
           break
    return scores
scores = cem()

        
def _evaluate(self, weights, gamma=1.0, max_t=5000):
    self.set_weights(weights)
    episode_return = 0.0
    state = self.env.reset()
    for t in range(max_t):
        state = torch.from_numpy(state).float().to(device)
        action = self.forward(state)
        state, reward, done, _ = self.env.step(action)
        episode_return += reward * math.pow(gamma, t)
        if done:
            break
    return episode_return
        
def _forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.tanh(self.fc2(x))
    return x.cpu().data
    
def _get_weights_dim(self):
    return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
        
        
def _set_weights(self, weights):
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
    










 


