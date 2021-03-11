import joblib
import tensorflow as tf
import numpy as np

# from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, multiply

from Agents import modelFreeAgent
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

tf.keras.backend.set_floatx('float64')

class DDPG(modelFreeAgent.ModelFreeAgent):
    displayName = 'DDPG'

    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True,
                                                             "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True,
                                                             "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True,
                                                             "The distance in timesteps between target model updates"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Tau', 0.00, 1.00, 0.001, 0.97, True, True,
                                                             "The rate at which target models update")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):

        # Initializing model parameters
        paramLen = len(DDPG.newParameters)
        super().__init__(*args[:-paramLen])

        self.batch_size, self.memory_size, self.target_update_interval, self.tau = [int(arg) for arg in args[-paramLen:]]
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        # self.ou_noise = OUNoise(self.action_size)

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)

    def get_actor(self):
        # Initialize actor network weights between -3e-3 and 3-e3
        last_in = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        input_shape = self.state_size
        inputA = Input(input_shape)
        x = Flatten()(inputA)
        x = Dense(24, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear', kernel_initializer=last_in)(x)
        model = Model(inputs=inputA, outputs=x)
        model.compile(loss='mse', optimizer=self.actor_optimizer)
        return model

    def get_critic(self):
        # Generate critic network model
        input_shape = self.state_size
        inputA = Input(shape=input_shape)
        inputB = Input(shape=(self.action_size,))
        x = Flatten()(inputA)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        model.compile(loss='mse', optimizer=self.critic_optimizer)
        return model

    def ou_noise(self, a, p=0.15, mu=0, differential=1e-1, sigma=0.2, dim=1):
        # Exploration noise generation
        return a + p * (mu - a) * differential + sigma * np.sqrt(differential) * np.random.normal(size=dim)

    def choose_action(self, state):

        bg_noise = np.zeros(self.action_size)
        bg_noise = self.ou_noise(bg_noise, dim=self.action_size)
        u = self.predict(state, False)
        sampled_actions = tf.squeeze(u)
        sampled_actions = sampled_actions.numpy() + bg_noise
        # Clipping action between bounds -0.3 and 0.3
        legal_action = np.clip(sampled_actions, -0.3, 0.3)[0]
        legal_action = np.squeeze(legal_action)
        action_returned = legal_action.astype(int)
        return action_returned

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):

        self.addToMemory(state, action, reward, new_state, done)
        loss = 0

        if len(self.memory) < 2*self.batch_size:
            return loss
        _, mini_batch = self.sample()
        X_train, Y_train, states = self.learn(mini_batch)

        # Train critic
        with tf.GradientTape() as tape:
            critic_value = self.critic_model([X_train])
            critic_loss = tf.math.reduce_mean(tf.math.square(Y_train - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        self.allBatchMask = self.allBatchMask.astype(float)
        actions = self.predict(states, False)
        actions = tf.convert_to_tensor(actions)
        o = self.critic_grads(states, self.allBatchMask)

        # Computing gradients using critic value"""
        critic_value = self.critic_model([states, self.allBatchMask], training=True)
        critic_value = tf.squeeze(critic_value)
        with tf.GradientTape() as tape:
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            # actor_loss = -tf.math.reduce_mean(critic_value)
            grad = tape.gradient(critic_value, actions)
        with tf.GradientTape() as tape:
            actor_grad = tape.gradient(self.predict(states, False), self.actor_model.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
        self.updateTarget()
        return critic_loss.numpy()

    def updateTarget(self):
        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            actor_weights = self.actor_model.get_weights()
            t_actor_weights = self.target_actor.get_weights()
            critic_weights = self.critic_model.get_weights()
            t_critic_weights = self.target_critic.get_weights()

            for i in range(len(actor_weights)):
                t_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * t_actor_weights[i]

            for i in range(len(critic_weights)):
                t_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * t_critic_weights[i]

            self.target_actor.set_weights(t_actor_weights)
            self.target_critic.set_weights(t_critic_weights)
            print("targets updated")
            self.total_steps += 1
        for ind in range(len(self.actor_model.get_weights())):
          self.target_actor.get_weights()[ind] = self.tau * self.actor_model.get_weights()[ind] + (1 - self.tau) * self.target_actor.get_weights()[ind]

        for ind in range(len(self.critic_model.get_weights())):
            self.target_critic.get_weights()[ind] = self.tau * self.critic_model.get_weights()[ind] + (1 - self.tau) * self.target_critic.get_weights()[ind]
        self.total_steps += 1


    def predict(self, state, isTarget):

        shape = (-1,) + self.state_size
        state = np.reshape(state, shape)
        # state = tf.cast(state, dtype=tf.float32)
        if isTarget:
            result = self.target_actor([state])
        else:
            result = self.actor_model([state])
        return result

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros(vector_length)
        if hot_index != -1:
            output[hot_index] = 1
        return output



    def learn(self, mini_batch):

        X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        states = (np.zeros((self.batch_size,) + self.state_size))
        next_states = (np.zeros((self.batch_size,) + self.state_size))


        for index_rep, transition in enumerate(mini_batch):
            X_train[0][index_rep] = transition.state
            states[index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))

        self.allBatchMask = self.allBatchMask.astype(float)

        qnext = self.target_critic([next_states, self.allBatchMask])
        qnext = np.amax(qnext, 1)
        for index_rep, transition in enumerate(mini_batch):
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma

        # c_loss = -tf.math.reduce_mean(critic_values)
        """c_loss = tf.math.reduce_mean(tf.math.square(Y_train - critic_values))
            grads = tape.gradient(c_loss, self.critic_model.trainable_variables)
        with tf.GradientTape() as tape:
            # actor_grad = tape.gradient(self.actor_model(X_train), self.actor_model.trainable_variables, grads)
            actor_grad = tape.gradient(self.actor_model(X_train), self.actor_model.trainable_variables, grads)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))"""

        """with tf.GradientTape() as tape:
            with tf.GradientTape() as tape:
                grads = tape.gradient(c_loss, self.critic_model.trainable_variables)
                # actor_grad = tape.gradient(self.actor_model(X_train), self.actor_model.trainable_variables, grads)
                actor_grad = tape.gradient(self.actor_model(X_train), self.actor_model.trainable_variables, grads)"""

        return X_train, Y_train, states

    def critic_grads(self, states, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            critic_value = self.critic_model([states, actions])
            critic_value = tf.squeeze(critic_value)
        return tape.gradient(critic_value, actions)

    def save(self, filename):
        act_mem = self.actor_model.get_weights()
        crit_mem = self.critic_model.get_weights()
        joblib.dump((DDPG.displayName, act_mem, crit_mem), filename)

    def load(self, filename):
        name, act_wt, crit_wt = joblib.load(filename)
        if name != DDPG.displayName:
            print('load failed')
        else:
            self.actor_model.set_weights(act_wt)
            self.critic_model.set_weights(crit_wt)

    def memsave(self):
        actor_weights = self.actor_model.get_weights()
        critic_weights = self.critic_model.get_weights()
        return (actor_weights, critic_weights)

    def memload(self, mem):
        act_wt, crit_wt = mem
        self.actor_model.set_weights(act_wt)
        self.critic_model.set_weights(crit_wt)

    def reset(self):
        pass
