from Agents import agent, ppo
from Agents.ppo import PPO
from Agents.deepQ import DeepQ
import tensorflow as tf
#from tensorflow.linalg.experimental import conjugate_gradient
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply
from tensorflow.keras import utils
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam

import numpy as np
import torch
import torch.nn as nn
import random
import joblib
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple

class TRPO(ppo.PPO):
    displayName = 'TRPO Agent'
    newParameters = ppo.PPO.newParameters
    parameters = ppo.PPO.parameters
    paramLen = len(newParameters)

    #Invoke constructor
    def __init__(self, *args):
        super().__init__(*args[:-paramLen])
        self.Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states',])
        self.rewards = []
        self.rollouts = []
        self.advantages = 0
    '''def optimize(self, action, state, policy, parameters, newParameters):
        advantage = 0
        for critic in range(10):
            for actor in range(10):
                # compute advantage
                advantage = 0
                super().updateActor()
            #optimize surrogate loss using recurrent neural network 
            parameters = newParameters
            super.updateCritic()'''

    '''def choose_action(self, state):
        observation = Tensor(state).unsqueeze(0)
        probabilities = super().criticModel(Variable(observation, requires_grad=True))
        action = probabilities.multimonial(1)
        return action, probabilities'''

    def remember(self, state, action, reward, new_state, done): 
        self.addToMemory(state, action, reward, new_state, done)
        loss = 0
        if len(self.memory) < 2*self.batch_size:
            return 0
        mini_batch = self.sample()
        num_rollouts = 10
        for t in range(num_rollouts): 
            DeepQ.reset()
            state = DeepQ.get_empty_state()
            done = False
            samples = []
            self.rewards = []
            while not done:
                with torch.no_grad():
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.action(state)

                    samples.append((state, action, reward, next_state))

                    state = next_state
                
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            self.rollouts.append(self.Rollout(states, actions, rewards, next_states))
        super().updateNetworks(mini_batch)
        self.updateAgent(self.rollouts)
        return self.surrogate_loss



    '''def train(self, num_rollouts=10):
        for epoch in range(super().epochSize):
            rollouts = []

            for t in range(num_rollouts):
                DeepQ.reset()
                state = DeepQ.get_empty_state()
                done = False

                samples = []
                self.rewards = []
                while not done:
                    with torch.no_grad():
                        action = self.choose_action(state)
                        next_state, reward, done, _ = self.action(state)

                        samples.append((state, action, reward, next_state))

                        state = next_state
                
                states, actions, rewards, next_states = zip(*samples)

                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
                actions = torch.as_tensor(actions).unsqueeze(1)
                rewards = torch.as_tensor(rewards).unsqueeze(1)

                rollouts.append(self.Rollout(states, actions, rewards, next_states))
        self.updateAgent(rollouts)
        return self.surrogate_loss '''

    def choose_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        dist = Categorical(super().actorModel(state))
        return dist.sample().item()

    def calculateTargetValues(self, mini_batch):
        pass

    def updateAgent(self, rollouts):
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        self.advantages = [self.estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
        self.advantages = torch.cat(self.advantages, dim=0).flatten()

        self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std()

        super().updateCritic(self.advantages)

        distribution = super().actorModel(states)

        distribution = torch.distributions.utils.clam_prob(distribution)

        probabilities = distribution[range(distribution.shape[0]), actions]

        self.L = self.surrogate_loss(probabilities, probabilities.detach())
        self.KL = self.kl_div(distribution, distribution)

        self.g = self.flat_grad(self.L, parameters, retain_graph=True)
        self.d_kl = self.flat_grad(self.KL, parameters, create_graph=True)

        def HVP(v):
            return self.flat_grad(self.d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, self.g)
        delta = 0.01
        max_length = torch.sqrt(2 * delta / (search_dir @ HVP(search_dir)))
        max_steps = max_length * search_dir

        def criterion(step):
            self.apply_update(step)

            with torch.no_grad():
                distribution_new = super().actorModel(states)
                distribution_new = torch.distributions.utils.clam_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

                self.L_new = self.surrogate_loss(probabilities_new, probabilities)
                self.KL_new = self.kl_div(distribution, distribution_new)

            L_improvement = self.L_new - self.L_improvement
            if L_improvement > 0 and self.KL_new <= delta:
                return True
        
            self.apply_update(-step)
            return False

            i = 0
            while not criterion((0.9 ** i) * max_steps) and i < 10:
                i += 1

    def estimate_advantages(self, states, last_state, rewards):
        values = super().criticModel(states)
        last_value = super().criticModel(last_state.unsqueeze(0))
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        self.advantages = next_values - values
        return self.advantages

    def surrogate_loss(self, new_probabilities, old_probabilities):
        return (new_probabilities / old_probabilities * self.advantages).mean()

    def kl_div(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()


    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    def conjugate_gradient(self, A, b, delta=0., max_iterations=10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        i = 0
        while i < max_iterations:
            AVP = A(p)

            dot_old = r @ r
            alpha = dot_old / (p @ AVP)

            x_new = x + alpha * p

            if (x - x_new).norm() <= delta:
                return x_new

            i += 1
            r = r - alpha * AVP

            beta = (r @ r) / dot_old
            p = r + beta * p

            x = x_new
        return x


    def apply_update(self, grad_flattened):
        n = 0
        for p in parameters:
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def update(self):
        self.train(num_rollouts=10)

    def save(self, filename):
        mem1 = super().actorMdel.get_weights()
        joblib.dump((TRPO.displayName, mem1), filename)
        mem2 = super().criticModel.get_weights
        joblib.dump((TRPO.displayName, mem2), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != TRPO.displayName:
            print('load failed')
        else:
            super().actorModel.set_weights(mem)
            super().actorTarget.set_weights(mem)
            super().criticModel.set_weights(mem)
            super().criticTarget.set_weights(mem)

    def memsave(self):
        return self.actorModel.get_weights(), self.criticModel.get_weights()

    def memload(self, mem):
        self.actorModel.set_weights(mem)
        self.actorTarget.set_weights(mem)
        self.criticModel.set_weights(mem)
        self.criticTarget.set_weights(mem)

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass
