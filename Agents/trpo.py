from Agents import agent, modelFreeAgent
from Agents.ppo import PPO
from Agents.deepQ import DeepQ
from Agents.models import Actor, Critic
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

import tensorflow as tf
#from tensorflow.linalg.experimental import conjugate_gradient
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply
from tensorflow.keras import utils
from tensorflow.keras.losses import KLDivergence, MSE
from tensorflow.keras.optimizers import Adam
#from tensorflow.train import GradientDescentOptimizer
#import tensorflow_probability as tfp

import numpy as np
import copy
import torch
import torch.nn as nn
import random
import joblib

from torch import Tensor
from torch.optim import Adam
from torch.distributions import Categorical
from torch.autograd import Variable

from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
#from utils.torch_utils import Tensor, Variable, ValueFunctionWrapper
#mport mathutils as math_utils

import collections
from collections import namedtuple


class TRPO(PPO):
    displayName = 'TRPO Agent'
    newParameters = [DeepQ.Parameter('Value learning rate+', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at")
                     ]
    parameters = PPO.parameters + newParameters

    #Invoke constructor
    def __init__(self, *args):
        print("Stuff TRPO:")
        print(str(args))
        paramLen = len(TRPO.newParameters)
        super().__init__(*args[:-paramLen])
        self.Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'new_states', 'action_dist'])
        self.epoches = 10
        self.gamma = 0.99
        '''self.min_epsilon = modelFreeAgent.ModelFreeAgent.min_epsilon
        self.max_epsilon = modelFreeAgent.ModelFreeAgent.max_epsilon
        self.decay_rate = modelFreeAgent.ModelFreeAgent.decay_rate
        self.time_steps = modelFreeAgent.ModelFreeAgent.time_steps'''

        self.parameters = TRPO.parameters
        self.newParameters = PPO.newParameters

        self.rollouts = []
        self.advantage = 0
        self.entropy = 0
        self.c1 = 0.001
        self.c2 = 0.001
        self.kl_penalty = 0.005
        self.residual_total = 1e-10
        self.loss = 0

        Qparams = []
        empty_state = self.get_empty_state()
        for i in range(3):
            Qparams.append(DeepQ.newParameters[i].default)
        self.batch_size, self.memory_size, self.target_update_interval = [int(param) for param in Qparams]
        
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)
        
        # Initialize the actors and critics
        self.value_model = super().value_network()
        self.policy_model = super().policy_network()
        self.actor_its = 10
        self.critic_its = 10

        print(self.policy_model.summary)
        print(self.value_model.summary)

    def get_empty_state(self):
        return super().get_empty_state()

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))
    
    def predict(self, state, isTarget):
        pass

    def remember(self, state, action, reward, new_state, done):
        self.addToMemory(state, action, reward, new_state, done)
        loss = 0
        if len(self.memory) < 2*self.batch_size:
            return loss
        mini_batch = self.sample()
        states, actions, rewards, new_states, old_probs = self.calculate_rollouts()
        # Create optimizer for minimizing loss
        optimizer = Adam(lr=self.policy_lr)

        # Compute old probability
        old_probs = old_probs.numpy()
        actions = actions.numpy()
        old_p = tf.math.log(tf.reduce_sum(np.multiply(old_probs, actions)))
        old_p = tf.stop_gradient(old_p)
        for i in range(self.critic_its):
            for j in range(self.actor_its):
                # Compute value estimates and advantage
                value_est = self.value_model.predict(states)
                value_est = np.average(value_est)
                value_next = self.value_model.predict(new_states)
                value_next = np.average(value_next)
                advantage = self.advantages(value_est, value_next)
                
                # Run the policy under N timesteps using loss function
                value_loss = self.c1 * self.mse_loss(states, new_states)
                clip_loss = self.clipped_loss(prob_ratio, advantage)
                for epoch in range(self.epoches):
                    self.train_policy(value_loss, clip_loss)
                
                # Compute new probabilities after training policy
                new_probs = self.policy_model.predict_proba(states, self.batch_size)
                new_probs = tf.convert_to_tensor(new_probs, dtype=tf.float32)
                new_probs = new_probs.numpy()
                new_p = tf.math.log(tf.reduce_sum(np.multiply(new_probs, actions)))
                
                # Compute probability ratio
                prob_ratio = tf.math.exp(new_p - old_p)

            # optimize loss function after training
            value_loss = self.c1 * self.mse_loss(states, new_states)
            clip_loss = self.clipped_loss(prob_ratio, advantage)
            loss = self.compute_loss(value_loss, clip_loss)
            # apply gradient optimizer to optimize loss
            with tf.GradientTape() as tape:
                loss = self.optimize_loss(loss, optimizer, tape)
            # update policy parameters
            self.parameters = self.policy_model.predict(self.parameters)
        return loss

    def choose_action(self, state):
        super().choose_action(state)

    def get_action(self, state, new_state):
        return super().get_action(state, new_state)

    def calculate_rollouts(self):
        self.entropy = 0
        transition = self.memory.peak_frame()
        states, actions, rewards, new_states, dones = transition
        state = transition.state
        new_state = transition.next_state
        action_dist = self.get_action(state, new_state)
        self.entropy += (action_dist * action_dist.log()).sum()
        self.entropy = self.entropy / len(actions)

        # Convert lists to numpy array 
        '''states = np.expand_dims(states, -1)
        actions = np.expand_dims(actions, -1)
        rewards = np.expand_dims(rewards, -1)
        new_states = np.expand_dims(new_states, -1)'''

        # Convert lists to tensor
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        probabilities = tf.convert_to_tensor(action_dist, dtype=tf.float32)

        self.rollouts.append(self.Rollout(states, actions, rewards, new_states, action_dist))
        return states, actions, rewards, new_states, probabilities
        #self.updateAgent(self.rollouts)
        #return self.loss

    def advantages(self, value_est, value_next):
        rewards = self.memory.get_recent_rewards()
        states = self.memory.get_recent_state()
        new_states = self.memory.get_recent_next_state()
        advantages = []
        for i in range (self.epoches):
            advantage = 0
            discounted_rewards = 0
            k = 0
            total_gamma = 0
            temp_states = []
            temp_new_states = []
            for j in range(self.epoches-i):
                total_gamma = self.gamma ** k
                discounted_rewards += total_gamma * rewards[j]
                k += 1
                temp_states = np.append(temp_states, states[j])
                temp_new_states = np.append(temp_new_states, new_states[j])

            #self.value_model.fit(temp_states, temp_new_states)
            v_est = self.value_model.predict(temp_states)
            advantage = -v_est + discounted_rewards + (total_gamma * v_est)
            advantages = np.append(advantages, advantage)
        mean = np.average(np.array(advantages))
        std = np.std(advantages)
        return (value_next - value_est - mean) / std

    def mse_loss(self, states, new_states):
        states_array = states.numpy()
        new_states_array = new_states.numpy()
        return self.value_model.evaluate(x=states_array, y=new_states_array, batch_size=self.batch_size)

    def clipped_loss(self, prob_ratio, advantage):
        #epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.time_steps)
        epsilon = 0.2
        minimum = tf.minimum(prob_ratio * advantage, 1 - epsilon, 1 + epsilon)
        loss = minimum * advantage
        return loss

    def compute_loss(self, value_loss, clip_loss):
        return clip_loss + value_loss + self.c2 * self.entropy

    def kl_divergence(self, states, new_states):
        kl = tf.keras.losses.KLDivergence()
        d_pred = self.policy_model.predict(states)
        d_true = self.policy_model.predict(new_states)
        return kl(new_states, states).numpy()

    def optimize_loss(self, loss, optimizer, tape):
        grads = tape.gradient(loss, self.parameters)
        optimizer.apply_gradients(zip(grads, self.parameters))
        return loss

    def train_policy(self, value_loss, clip_loss):
        with tf.GradientTape() as tape:
            self.policy_model(self.parameters, training=true)
            loss = self.compute_loss(value_loss, clip_loss)
            tape.gradient(loss, self.parameters)


    '''def policy_gradient(self, loss):
        self.policy_network.zero_grad()
        loss.backward(retain_graph=True)
        policy_gradient = parameters_to_vector([v.grad for v in self.policy_network.parameters()]).squeeze(0)
        return policy_gradient'''
    '''def updateAgent(self, rollouts):
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0)
        rewards = torch.cat([r.rewards for r in rollouts], dim=0).flatten()
        new_states = torch.cat([r.new_states for r in rollouts], dim=0)
        probabilities = torch.cat([r.action_dist for r in rollouts], dim=0)

        # Compute advantage function used for computing loss function.
        baseline = self.value_network.predict(states).data
        rewards_tensor = Tensor(rewards).unsqueeze(1)
        advantage = rewards_tensor - baseline

        # Normalize the advantage
        self.advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        loss = self.initial_loss(actions, probabilities)
        policy_gradient = self.policy_gradient(loss)

        if (policy_gradient.nonzero().size()[0]):
            step_dir = self.conjugate_gradient(-policy_gradient)
            dir_variable = Variable(torch.from_numpy(step_dir))

        shs = .5 * step_dir.dot(self.hessian_vector(dir_variable).cpu().numpy().t)
        lm = np.sqrt(shs / self.max_kl)
        fullstep = step_dir / lm
        grad_step = -policy_gradient.dot(dir_variable).data[0]
        theta = self.linesearch(parameters_to_vector(self.policy_network.parameters()), states, actions, fullstep, grad_step / lm)

        # Fit estimated value function to actual rewards
        #ev_before = math_utils.explained_variance_1d(baseline.squeeze(1).cpu().numpy(), rewards)
        self.value_network.zero_grad()
        value_params = parameters_to_vector(self.value_network.parameters())
        self.value_network.fit(states, Variable(Tensor(rewards)))
        #ev_after = math_utils.explained_variance_1d(self.value_network.predict(self.states).data.squeeze(1).cpu().numpy(), rewards)

        #if ev_after < ev_before or np.abs(ev_after) < 1e-4:
        vector_to_parameters(value_params, self.value_network.parameters())


    def initial_loss(self, actions, probabilities):
        prob_new = torch.cat(probabilities).gather(1, torch.cat(actions))
        prob_old = prob_new.detach() + 1e-8
        prob_ratio = prob_new / prob_old
        loss = -torch.mean(prob_ratio * Variable(self.advantage)) - (self.ent_coeff * self.entropy)
        return loss


    def kl_divergence(self, model, states):
        states_tensor = torch.cat([Variable(Tensor(state)).unsqueeze(0) for state in states])
        action_prob = model(states_tensor).detach() + 1e-8
        old_action_prob = self.policy_network(states_tensor)
        return torch.sum(old_action_prob * torch.log(old_action_prob/action_prob), 1).mean()

    # Computes hessian vector product

    def hessian_vector(self, vector):
        self.policy_network.zero_grad()
        kl_div = self.kl_divergence(self.policy_network)
        kl_grad = torch.autograd.grad(kl_div, self.policy_network.parameters(), create_graph=True)
        kl_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        v_product = torch.sum(kl_vector * vector)
        grad_product = torch.autograd.grad(v_product, self.policy_network.parameters())
        actual_product = torch.cat([grad.contiguous().view(-1) for grad in grad_product]).data
        return actual_product + (self.damping * vector.data)


    # Uses conjugate gradient to ensure policy upates aren't too big or too small.
    def conjugate_gradient(self, b, max_iterations=10):
        r = b.clone().data
        p = b.clone().data
        x = np.zeros_like(b.data.cpu().numpy())
        r_dotr = r.double().dot(r.double())
        for i in range(max_iterations):
            z = self.hessian_vector(Variable(p)).squeeze(0)
            v = r_dotr / p.double().dot(z.double())
            x += v * p.cpu().numpy()
            r -= v * z
            newr_dotr = r.double().dot(r.double())
            mu = newr_dotr / r_dotr
            p = r + mu * p
            r_dotr = newr_dotr
            if r_dotr < self.residual_total: 
                break
        return x

    def surrogate_loss(self, theta, states, actions):
        new_policy_network = copy.deepcopy(self.policy_network)
        vector_to_parameters(theta, new_policy_network.parameters())
        states_tensor = torch.cat([Variable(Tensor(state)).unsqueeze(0) for state in states])
        new_prob = new_policy_network(states_tensor).gather(1, torch.cat(actions)).data
        old_prob = self.policy_network(states_tensor).gather(1, torch.cat(self.actions)).data + 1e-8
        return -torch.mean((new_prob / old_prob) * self.advantage)

    def linesearch(self, states, actions, x, fullstep, exp_improverate):
        accept_ratio = .1
        max_backtracks = 10
        fval = self.surrogate_loss(x, states, actions)
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            xnew = x.data.cpu().numpy() + stepfrac * fullstep
            newfval = self.surrogate_loss(Variable(torch.from_numpy(xnew)), states, actions)
            actual_improvement = fval - newfval
            expected_improvement = exp_improverate * stepfrac
            ratio = actual_improvement / expected_improvement
            if ratio > accept_ratio and actual_improvement > 0:
                self.loss = newfval
                return Variable(torch.from_numpy(xnew))
        self.loss = fval
        return x        

    # Performs a policy update by updating parameters. 
    def apply_update(self, grad_flattened):
        n = 0
        for p in TRPO.parameters:
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel'''

    def update(self):
        pass

    def save(self, filename):
        mem1 = self.value_model.get_weights()
        joblib.dump((TRPO.displayName, mem1), filename)
        mem2 = self.policy_model.get_weights()
        joblib.dump((TRPO.displayName, mem2), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != TRPO.displayName:
            print('load failed')
        else:
            self.policy_model.set_weights(mem)
            self.value_model.set_weights(mem)

    def memsave(self):
        return self.policy_model.get_weights()

    def memload(self, mem):
        self.policy_model.set_weights(mem)
        self.value_model.set_weights(mem)

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass
