
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
from tensorflow.compat.v1.train import GradientDescentOptimizer
#import tensorflow_probability as tfp

import numpy as np
import copy
import random
import joblib
import math

class TRPO(PPO):
    displayName = 'TRPO'
    newParameters = [DeepQ.Parameter('Value learning rate+', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at")
                     ]
    parameters = PPO.parameters + newParameters

    #Invoke constructor
    def __init__(self, *args):
        paramLen = len(TRPO.newParameters)
        super().__init__(*args[:-paramLen])
        self.gamma = 0.99
        '''self.min_epsilon = modelFreeAgent.ModelFreeAgent.min_epsilon
        self.max_epsilon = modelFreeAgent.ModelFreeAgent.max_epsilon
        self.decay_rate = modelFreeAgent.ModelFreeAgent.decay_rate
        self.time_steps = modelFreeAgent.ModelFreeAgent.time_steps'''

        self.parameters = TRPO.parameters
        self.newParameters = PPO.newParameters

        self.c1 = 0.001
        self.c2 = 0.001
        self.loss = 0
        self.Lambda = 1

        '''Qparams = []
        empty_state = self.get_empty_state()
        for i in range(3):
            Qparams.append(DeepQ.newParameters[i].default)
        self.batch_size, self.memory_size, self.target_update_interval = [int(param) for param in Qparams]
        
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))'''
        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.actorIts, self.action_size), 1)

    def sample(self):
        return self.memory.sample(self.actorIts)

    def addToMemory(self, state, action, reward, next_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, next_state, done))
    
    def predict(self, state, isTarget):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.addToMemory(state, action, reward, next_state, done)
        loss = 0
        if len(self.memory) < 2*self.batch_size:
            return loss
        losses = []
        _, mini_batch = self.sample()
        states, actions, rewards, next_states, dones = self.sample_trajectories(mini_batch)
        X_train, Y_train = self.calculateTargetValues(mini_batch)
        self.value_model.train_on_batch(X_train, Y_train)
        # Create optimizer for minimizing loss
        optimizer = GradientDescentOptimizer(learning_rate= 0.001)
        for idx, transition in enumerate(mini_batch):
            state = transition.state 
            next_state = transition.next_state
            shape = (1,) + self.state_size
            state = np.reshape(state, shape)
            next_state = np.reshape(next_state, shape)
            # Compute old probability
            old_probs = self.get_probabilities(states)
            old_probs = np.array(old_probs)
            actions = np.array(actions)
            old_p = tf.math.log(tf.reduce_sum(np.multiply(old_probs, actions)))
            old_p = tf.stop_gradient(old_p)
            goal = self.goal_idx(idx)
            # Compute advantage
            advantage = self.get_advantages(idx, goal)
            # Compute new probabilities 
            new_probs = self.policy_model([states, self.allBatchMask], training=True)
            new_probs = np.array(new_probs)
            new_p = tf.math.log(tf.reduce_sum(np.multiply(new_probs, actions)))
            # Compute probability ratio
            prob_ratio = tf.math.exp(new_p - old_p)
            # Run the policy under N timesteps using loss function
            value_loss = self.c1 * self.mse_loss(state, next_state)
            clip_loss = self.clipped_loss(prob_ratio, advantage)
            self.train_policy(states, clip_loss)
            entropy = self.get_entropy(state)
            loss = self.agent_loss(value_loss, clip_loss, entropy)
            losses.append(loss)
        loss = np.mean(np.array(losses))
        print("loss iteration: " + str(loss))
        self.updateTarget()
        # apply gradient optimizer to optimize loss and policy network
        '''with tf.GradientTape() as tape:
            loss = self.optimize_loss(loss, optimizer, tape)'''
        return loss

    def sample_trajectories(self, mini_batch):
        states = np.zeros(((self.actorIts, ) + self.state_size))
        next_states = np.zeros(((self.actorIts, ) + self.state_size))
        actions = np.zeros(((self.actorIts, ) + (self.action_size, )))
        rewards = []
        dones = []
        for index, transition in enumerate(mini_batch):
            state, action, reward, next_state, done = transition
            states[index, :] = state
            actions[index, :] = action
            rewards.append(reward)
            next_states[index, :] = next_state
            dones.append(done)
        return states, actions, rewards, next_states, dones

    def goal_idx(self, idx):
        transitions = self.memory._transitions
        while idx < len(self.memory._transitions)-1 and transitions[idx] is not None and transitions[idx].is_done is False:
            idx+=1
        return idx

    '''def update_policy(self):
        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.policy_model.set_weights(self.newParameters)
            print("target updated")
        self.total_steps += 1'''

    def updateTarget(self):
        self.total_steps+=1

    def calculateTargetValues(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        next_states = np.zeros((self.actorIts,) + self.state_size)

        for index_rep, transition in enumerate(mini_batch):
            states, actions, rewards, _, dones = transition
            
            X_train[0][index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        vnext = self.value_model.predict([next_states, self.allBatchMask])
        vnext = np.mean(vnext, axis=1)
        for index_rep, transition in enumerate(mini_batch):
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + vnext[index_rep] * (self.gamma ** (self.batch_size-index_rep))
        return X_train, Y_train


    def get_advantages(self, idx, goal):
        #print("Goal: " + str(goal))
        transitions = self.memory.get_next_transitions(idx, goal)
        states = [transitions[i].state for i in range(goal-idx)]
        next_states = [transitions[i].next_state for i in range(goal-idx)]
        rewards = [transitions[i].reward for i in range(goal-idx)]
        advantages = []
        advantage = 0
        total_gamma = 0
        for j in range(goal-idx):
            total_gamma = (self.gamma * self.Lambda) ** j
            # discouNnted_rewards += total_gamma * rewards[j]
            shape = (1,) + self.state_size
            state = np.reshape(states[j], shape)
            next_state = np.reshape(next_states[j], shape)
            v = self.value_model.predict([state, self.allMask])
            v = np.mean(v)
            v_next = self.value_model.predict([next_state, self.allMask])
            v_next = np.mean(v_next)
            advantage += (total_gamma * v_next) - v + rewards[j]
        advantages.append(advantage)
        mean = np.mean(np.array(advantages), axis=0)
        low = np.min(np.array(advantages))
        high = np.max(np.array(advantages))
        if high - low == 0:
            return 0
        return (mean - low) / (high - low)

    def mse_loss(self, state, next_state):
        v_pred = self.value_model.predict([state, self.allMask])
        v_true = self.value_model.predict([next_state, self.allMask])
        #return self.value_model.evaluate(x=[states, self.allBatchMask], y=[next_states, self.allBatchMask], batch_size=self.batch_size)
        return tf.reduce_sum(MSE(v_true, v_pred)).numpy()

    def clipped_loss(self, prob_ratio, advantage):
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.time_steps)
        #epsilon = 0.2
        clips = []
        adv_ratio = (prob_ratio*advantage).numpy()
        adv_ratio = np.mean(adv_ratio)
        clips.append(adv_ratio)
        clips.append(1-epsilon)
        clips.append(1+epsilon)
        minimum = math.inf
        for clip in clips:
            minimum = min(minimum, clip)
        loss = minimum * advantage
        return loss

    def get_entropy(self, state):
        from scipy.stats import entropy
        shape = (1,) + self.state_size
        state = np.reshape(state, shape)
        probabilities = self.policy_model.predict([state, self.allMask])
        #return tf.math.log(tf.reduce_sum((np.array(entropy(probabilities)))))
        value, counts = np.unique(probabilities, return_counts=True)
        probs = counts / len(probabilities)
        entropy = 0
        for i in probs:
            entropy -= i * math.log(i, math.e)
        return entropy

    def agent_loss(self, value_loss, clip_loss, entropy):
        return clip_loss + value_loss + (self.c2 * entropy)

    '''def kl_divergence(self, states, new_states):
        kl = tf.keras.losses.KLDivergence()
        d_pred = self.policy_model.predict(states)
        d_true = self.policy_model.predict(new_states)
        return kl(new_states, states).numpy()'''

    def optimize_loss(self, loss, optimizer, tape):
        loss = np.array(loss)
        loss = tf.convert_to_tensor(loss)
        grads = tape.gradient(loss, self.policy_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
        return loss

    def train_policy(self, states, clip_loss):
        with tf.GradientTape() as tape:
            tape.watch(self.policy_model.trainable_variables)
            self.policy_model([states, self.allBatchMask], training=True)
            #loss = np.array(self.agent_loss(value_loss, clip_loss, entropy))
            #loss = tf.convert_to_tensor(loss)
            #loss = self.agent_loss(value_loss, clip_loss, entropy)
            loss = tf.convert_to_tensor(clip_loss)
            tape.gradient(loss, self.policy_model.trainable_variables)

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
