import joblib
import tensorflow as tf
from typing import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, multiply
import numpy as np

from Agents import modelFreeAgent
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

tf.keras.backend.set_floatx('float64')

# Class to create Actor Network
class actorNetwork(Model):
    """
    Source: https://github.com/RickyMexx/SAC-tf2/blob/master/common/utils.py.
    """
    def __init__(self, action_dim):
        super(actorNetwork, self).__init__()
        self.logprob_epsilon = 1e-6
        actor_bound = 3e-3
        self.network = Sequential()
        for i in range(2):
            self.network.add(Dense(24, activation="relu"))

        self.mean = Dense(action_dim,
                          kernel_initializer=tf.random_uniform_initializer(-actor_bound, actor_bound),
                          bias_initializer=tf.random_uniform_initializer(-actor_bound, actor_bound))
        self.prob = Dense(action_dim,
                             kernel_initializer=tf.random_uniform_initializer(-actor_bound, actor_bound),
                             bias_initializer=tf.random_uniform_initializer(-actor_bound, actor_bound))

    @tf.function
    def call(self, inp):
        x = self.network(inp)
        mean = self.mean(x)
        log_std = self.prob(x)
        prob_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tf.compat.v1.distributions.Normal(mean, tf.exp(prob_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        action_returned = tf.tanh(action)
        prob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(action_returned, 2) + self.logprob_epsilon)
        prob = tf.reduce_sum(prob, axis=-1, keepdims=True)
        return action_returned, prob

    def _get_params(self):
        with self.graph.as_default():
            parameters = tf.trainable_variables()
        name = [s.name for s in parameters]
        value_return = self.sess.run(parameters)
        params = {k: v for k, v in zip(name, value_return)}
        return params

    def __getstate__(self):
        params = self._get_params()
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        self.restore_params(params)


def soft_update(source: Sequence[tf.Variable], target: Sequence[tf.Variable], tau: float):
    if len(source) != len(target):
        raise ValueError("source_vars and target_vars must have the same length.")
    for source, target in zip(source, target):
        target.assign((1.0 - tau) * target + tau * source)
    return target


def force_update(source: Sequence[tf.Variable], target: Sequence[tf.Variable]):
    soft_update(source, target, 1.0)


class SAC(modelFreeAgent.ModelFreeAgent):
    displayName = 'SAC'

    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True,
                                                             "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True,
                                                             "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True,
                                                             "The distance in timesteps between target model updates"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Tau', 0.00, 1.00, 0.001, 0.97, True, True,
                                                             "The rate at which target models update"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Temperature', 0.00, 1.00, 0.001, 0.97, True, True,
                                                             "The rate at which target models update")
                     ]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):

        # Initializing model parameters
        paramLen = len(SAC.newParameters)
        super().__init__(*args[:-paramLen])

        self.batch_size, self.memory_size, self.target_update_interval, self.tau, self.temperature = [int(arg) for arg
                                                                                                      in
                                                                                                      args[-paramLen:]]
        self.polyak = 0.01
        self.total_steps = 0

        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size,
                                                    TransitionFrame(empty_state, -1, 0, empty_state, False))
        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.actor_network = actorNetwork(self.action_size)

        self.soft_Q_network = self.q_network()
        self.soft_Q_targetnetwork = self.q_network()

        self.soft_Q_network1 = self.q_network()
        self.soft_Q_targetnetwork1 = self.q_network()

        # Building up 2 soft q-function with their relative targets
        in1 = tf.keras.Input(shape=self.state_size, dtype=tf.float64)
        in2 = tf.keras.Input(shape=self.action_size, dtype=tf.float64)

        self.soft_Q_network([in1, in2])
        self.soft_Q_targetnetwork([in1, in2])
        force_update(self.soft_Q_network.variables, self.soft_Q_targetnetwork.variables)

        self.soft_Q_network1([in1, in2])
        self.soft_Q_targetnetwork1([in1, in2])
        force_update(self.soft_Q_network1.variables, self.soft_Q_targetnetwork1.variables)

        # Optimizers for the networks
        self.softq_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.softq_optimizer2 = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    def q_network(self):
        # Generate critic network model
        input_shape = self.state_size
        inputA = Input(shape=input_shape)
        inputB = Input(shape=(self.action_size,))
        x = Flatten()(inputA)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        model.compile(loss='mse', optimizer=self.critic_optimizer)
        return model

    def value_network(self):
        # Generate critic network model
        input_shape = self.state_size
        inputA = Input(shape=input_shape)
        x = Flatten()(inputA)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=[inputA], outputs=x)
        model.compile(loss='mse', optimizer=self.critic_optimizer)
        return model

    def soft_q_value(self, states: np.ndarray, actions: np.ndarray):
        return self.soft_Q_network(states, actions)

    def soft_q_value1(self, states: np.ndarray, actions: np.ndarray):
        return self.soft_Q_network1(states, actions)

    def action(self, states):
        """Get action for a state."""
        return self.actor_network(states)[0][0]

    def actions(self, states):
        """Get actions for a batch of states."""
        return self.actor_network(states)[0]

    def choose_action(self, state):
        """Get the action for a single state."""

        shape = (-1,) + self.state_size
        state = np.reshape(state, shape)
        u = self.action(state)
        action_returned = u.numpy()[0]
        action_returned = action_returned.astype(int)
        return action_returned

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):

        self.addToMemory(state, action, reward, new_state, done)
        loss = 0

        if len(self.memory) < 2 * self.batch_size:
            return loss
        _, mini_batch = self.sample()
        states, actions, next_states, rewards, dones = self.learn(mini_batch)
        states = states.astype(float)
        next_states = next_states.astype(float)

        # Evaluating action probability of doing an action
        action, action_prob = self.actor_network(states)

        val_target = self.soft_Q_network([next_states, actions])
        val_target1 = self.soft_Q_network1([next_states, actions])

        # Minimizing values
        nextval_sample = tf.math.minimum(val_target, val_target1) - self.temperature * action_prob

        # Getting Q function targets
        Q_targets = rewards + self.gamma * (1 - dones) * nextval_sample
        # softq_targets = tf.reshape(softq_targets, [self.batch_size, 1])

        # Gradient descent for Q function - 1 and computing gradients
        with tf.GradientTape() as qtape:
            Q = self.soft_Q_network([states, actions])
            Q_loss= tf.reduce_mean(tf.square(Q - Q_targets))
        softq_gradients = qtape.gradient(Q_loss, self.soft_Q_network.trainable_weights)

        # Gradient descent for Q function - 2 and computing gradients
        with tf.GradientTape() as qtape2:
            Q2 = self.soft_Q_network1([states, actions])
            Q2_loss = tf.reduce_mean(tf.square(Q2 - Q_targets))
        softq_gradients2 = qtape2.gradient(Q2_loss, self.soft_Q_network1.trainable_weights)

        # Gradient ascent for policy and computing gradients
        with tf.GradientTape() as tape:
            # actions = self.actorModel()
            actions, action_logprob = self.actor_network(states)
            soft_Q = tf.math.minimum(self.soft_Q_network([states, actions]), self.soft_Q_network1([states, actions]))

            # Calculating loss
            loss_policy = tf.reduce_mean(action_logprob - soft_Q)
        actor_gradients = tape.gradient(loss_policy, self.actor_network.trainable_weights)

        # Apply gradients
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_weights))
        self.softq_optimizer.apply_gradients(zip(softq_gradients, self.soft_Q_network.trainable_weights))
        self.softq_optimizer2.apply_gradients(zip(softq_gradients2, self.soft_Q_network1.trainable_weights))

        Q_loss = Q_loss.numpy()
        self.updateTarget()
        return Q_loss

    def updateTarget(self):
        if self.total_steps >= 2 * self.batch_size and self.total_steps % self.target_update_interval == 0:
            # Update the weights of target networks
            soft_update(self.soft_Q_network.variables, self.soft_Q_targetnetwork.variables, self.polyak)
            soft_update(self.soft_Q_network1.variables, self.soft_Q_targetnetwork1.variables, self.polyak)
            print("targets updated")
        self.total_steps += 1

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros(vector_length)
        if hot_index != -1:
            output[hot_index] = 1
        return output

    def learn(self, mini_batch):
        states = (np.zeros((self.batch_size,) + self.state_size))
        actions = np.zeros((self.batch_size,) + (self.action_size,))
        next_states = (np.zeros((self.batch_size,) + self.state_size))
        rewards = np.zeros((self.batch_size,) + (self.action_size,))
        dones = np.zeros((self.batch_size,) + (self.action_size,))

        for index_rep, transition in enumerate(mini_batch):
            states[index_rep] = transition.state
            actions[index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state
            rewards[index_rep] = transition.reward
            dones[index_rep] = transition.is_done

        return states, actions, next_states, rewards, dones

    def predict(self, state, isTarget):

        shape = (-1,) + self.state_size
        state = np.reshape(state, shape)
        # state = state(float)

        if isTarget:
            print("Target achieved")
        else:
            result1 = self.action(state)

        return result1

    def save(self, filename):
        self.actor_network(np.reshape(self.get_empty_state(), (-1,) + self.state_size))
        act_mem = self.actor_network.get_weights()
        s0_mem = self.soft_Q_targetnetwork.get_weights()
        s1_mem = self.soft_Q_targetnetwork1.get_weights()
        mem = self.actor_network.get_weights()
        joblib.dump((SAC.displayName, act_mem, s0_mem, s1_mem), filename)
        print('Model saved')

    def load(self, filename):
        name, act_wt, s0_wt, s1_wt = joblib.load(filename)
        if name != SAC.displayName:
            print('load failed')
        else:
            self.actor_network(np.reshape(self.get_empty_state(), (-1,) + self.state_size))
            self.actor_network.set_weights(act_wt)
            self.soft_Q_targetnetwork.set_weights(s0_wt)
            self.soft_Q_targetnetwork1.set_weights(s1_wt)


    def memsave(self):
        self.actor_network(np.reshape(self.get_empty_state(), (-1,) + self.state_size))
        actor_weights = self.actor_network.get_weights()
        soft0_weights = self.soft_Q_targetnetwork.get_weights()
        soft1_weights = self.soft_Q_targetnetwork1.get_weights()
        return (actor_weights, soft0_weights, soft1_weights)

    def memload(self, mem):
        act_wt, s0_wt, s1_wt = mem
        self.actor_network(np.reshape(self.get_empty_state(), (-1,) + self.state_size))
        self.actor_network.set_weights(act_wt)
        self.soft_Q_targetnetwork.set_weights(s0_wt)
        self.soft_Q_targetnetwork1.set_weights(s1_wt)

    def reset(self):
        pass
