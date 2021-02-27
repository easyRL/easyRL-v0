from Agents import agent, modelFreeAgent
from Agents.ppo import PPO
from Agents.deepQ import DeepQ
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

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

class TRPO(PPO):
<<<<<<< HEAD
    displayName = 'TRPO Agent'
    newParameters = [DeepQ.Parameter('Value learning rate+', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at")
                     ]
=======
    displayName = 'TRPO'
    newParameters = [PPO.Parameter('Value learning rate+', 0.00001, 1, 0.00001, 0.001, True, True, "A learning rate that the Adam optimizer starts at")]
>>>>>>> 1b51932679684081c2d16e439b04d8a8765667e4
    parameters = PPO.parameters + newParameters

    #Invoke constructor
    def __init__(self, *args):
        print("Stuff TRPO:")
        print(str(args))
        paramLen = len(TRPO.newParameters)
        super().__init__(*args[:-paramLen])
        self.Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states',])
        self.rewards = []
        self.rollouts = []
        self.advantages = 0

        Qparams = []
        for i in range(3):
            Qparams.append(DeepQ.newParameters[i].default)
        self.batch_size, self.memory_size, self.target_update_interval = [int(param) for param in Qparams]

        self.actorModel = super().buildActorNetwork()
        self.actorTarget = super().buildActorNetwork()
        self.criticModel = super().buildCriticNetwork()
        self.criticTarget = super().buildCriticNetwork()

    def get_empty_state(self):
        return super().get_empty_state()

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done):
        self.addToMemory(state, action, reward, new_state, done)
        surrogate_loss = 0
        if len(self.memory) < 2*self.batch_size:
            return surrogate_loss
        mini_batch = self.sample()
        surrogate_loss = self.calculateTargetValues(mini_batch)
        return surrogate_loss

    '''def choose_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        dist = Categorical(self.actorModel(state, self.action_size))
        return dist.sample().item()'''


    '''def choose_action(self, state):
        val = self.predict(state, False)
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.time_steps)
        # TODO: Put epsilon at a level near this
        #if random.random() > epsilon:
        action = np.argmax(val)
        #else:
        #    action = self.state_size.sample()
        return action'''

    def calculateTargetValues(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        next_states = np.zeros((self.batch_size,) + self.state_size)
        self.rewards = []
        for index_rep, transition in enumerate(mini_batch):
            print("length of mini_batch: " + str(len(mini_batch[0])))
            print("MINI BATCH: \n")
            for i in range(len(mini_batch)):
                print("length of batch: " + str(len(mini_batch[i])))
            states, actions, rewards, next_state, dones = mini_batch
            X_train[0][index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)
            self.rollouts.append(self.Rollout(states, actions, rewards, next_states))
        self.updateAgent(self.rollouts)
        super().updateNetworks(mini_batch)
        return self.surr_new


    def updateAgent(self, rollouts):
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        # Compute advantage function used for computing loss function.
        self.advantages = [self.estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
        self.advantages = torch.cat(self.advantages, dim=0).flatten()

        self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std()

        super().updateCritic(self.advantages)

        # Computes distribution and performs importance sampling.
        # Computes distribution of dataset to use for calculating probabiities.
        distribution = self.actorModel(states)

        distribution = torch.distributions.utils.clam_prob(distribution)

        # Computes probabilities via Importance Sampling in order to compute loss function.
        probabilities = distribution[range(distribution.shape[0]), actions]

        self.surr_loss = self.surrogate_loss(probabilities, probabilities.detach())
        self.KL = self.kl_div(distribution, distribution)

        self.g = self.flat_grad(self.L, TRPO.parameters, retain_graph=True)
        self.d_kl = self.flat_grad(self.KL, TRPO.parameters, create_graph=True)

        def HVP(v):
            return self.flat_grad(self.d_kl @ v, TRPO.parameters, retain_graph=True)

        # Compute search direction in the gradient.
        search_dir = self.conjugate_gradient(HVP, self.g)
        delta = 0.01
        max_length = torch.sqrt(2 * delta / (search_dir @ HVP(search_dir)))
        max_steps = max_length * search_dir

        # Defines the size of policy update and computes loss.
        def criterion(step):
            self.apply_update(step)

            with torch.no_grad():
                distribution_new = self.actorModel(states)
                distribution_new = torch.distributions.utils.clam_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

                self.surr_new = self.surrogate_loss(probabilities_new, probabilities)
                self.KL_new = self.kl_div(distribution, distribution_new)

            L_improvement = self.surr_new - self.surr_loss
            if L_improvement > 0 and self.KL_new <= delta:
                return True
        
            self.apply_update(-step)
            return False

            i = 0
            while not criterion((0.9 ** i) * max_steps) and i < 10:
                i += 1

    # Estimate the advantages used for advantage function.
    def estimate_advantages(self, states, last_state, rewards):
        values = self.criticModel(states)
        last_value = PPO.criticModel(last_state.unsqueeze(0))
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        self.advantages = next_values - values
        return self.advantages

    # Uses probabilities and advantage value to compute surrogate loss
    # Calculates the mean for expected value
    def surrogate_loss(self, new_probabilities, old_probabilities):
        return (new_probabilities / old_probabilities * self.advantages).mean()

    def kl_div(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()

    # Performs grid search.
    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    # Uses conjugate gradient to ensure policy upates aren't too big or too small.
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

    # Performs a policy update by updating parameters. 
    def apply_update(self, grad_flattened):
        n = 0
        for p in TRPO.parameters:
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def update(self):
        pass

    def save(self, filename):
        mem1 = self.actorModel.get_weights()
        joblib.dump((TRPO.displayName, mem1), filename)
        mem2 = self.criticModel.get_weights()
        joblib.dump((TRPO.displayName, mem2), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != TRPO.displayName:
            print('load failed')
        else:
            self.actorModel.set_weights(mem)
            self.actorTarget.set_weights(mem)
            self.criticModel.set_weights(mem)
            self.criticTarget.set_weights(mem)

    def memsave(self):
        return self.criticModel.get_weights()

    def memload(self, mem):
        self.actorModel.set_weights(mem)
        self.actorTarget.set_weights(mem)
        self.criticModel.set_weights(mem)
        self.criticTarget.set_weights(mem)

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass
