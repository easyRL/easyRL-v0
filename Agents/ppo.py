from Agents import agent, modelFreeAgent
from Agents.deepQ import DeepQ
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply
from tensorflow.keras import utils
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam

class PPO:

    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Policy learning rate', 0.00001, 1, 0.00001, 0.001, True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Value learning rate', 0.00001, 1, 0.00001, 0.001,
                                                             True, True,
                                                             "A learning rate that the Adam optimizer starts at"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Horizon', 10, 10000, 1, 50,
                                                             True, True,
                                                             "The number of timesteps over which the returns are calculated"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Epoch Size', 10, 100000, 1, 500,
                                                             True, True,
                                                             "The length of each epoch (likely should be the same as the max episode length)"),
                     modelFreeAgent.ModelFreeAgent.Parameter('PPO Epsilon', 0.00001, 0.5, 0.00001, 0.2,
                                                             True, True,
                                                             "A measure of how much a policy can change w.r.t. the states it's trained on"),
                     modelFreeAgent.ModelFreeAgent.Parameter('PPO Lambda', 0.5, 1, 0.001, 0.95,
                                                             True, True,
                                                             "A parameter that when set below 1, can decrease variance while maintaining reasonable bias")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(PPO.newParameters)
        super().__init__(*args[:-paramLen])
        empty_state = self.get_empty_state()
        # Initialize parameters
        self.memory_size = DeepQ.memory_size
        self.batch_size = DeepQ.batch_size
        self.target_update_interval = DeepQ.target_update_interval
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)
        '''self.action_size = DeepQ.action_size
        self.state_size = DeepQ.state_size'''
        #self.batch_size, _, _, self.horizon, self.epochSize, _, _ = [int(arg) for arg in args[-paramLen:]]
        #_, self.policy_lr, self.value_lr, _, _, self.epsilon, self.lam = [arg for arg in args[-paramLen:]]
        '''self.mini_batch = mini_batch
        self.gamma = gamma
        self.horizon = horizon
        self.epoch = epoch
        self.episodes = episodes
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.rewards = []'''

        # Set up actor neural network
        self.actorModel = self.buildActorNetwork()
        self.actorTarget = self.buildActorNetwork()
        
        # Set up critic neural network
        self.criticModel = self.buildCriticNetwork()
        self.criticTarget = self.buildCriticNetwork()

    def buildActorNetwork(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(self.state_size, 32),
                              nn.ReLU(),
                              nn.Linear(32, self.action_size),
                              nn.Softmax(dim=1))
        #model.compile(loss='mse', optimizer=Adam(lr=self.value_lr, clipvalue=1), metrics=[metrics.mean_squared_error], steps_per_execution=10)
        '''inputA = Input(shape=self.state_size)
        inputB = Input(shape=(self.action_size,))

        if len(self.state_size) == 1:
            x = TimeDistributed(Dense(10, input_shape=self.state_size, activation='relu'))(inputA)
        else:
            x = TimeDistributed(Conv2D(16, 8, strides=4, activation='relu'))(inputA)
            x = TimeDistributed(Conv2D(32, 4, strides=2, activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(256)(x)
        x = Dense(10, activation='relu')(x)  # fully connected
        x = Dense(10, activation='relu')(x)
        x = Dense(self.action_size)(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)'''
        #model.compile(loss=kl, optimizer=Adam(lr=0.0001, clipvalue=1))
        return model

    def buildCriticNetwork(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(self.state_size, 32),
                              nn.ReLU(),
                              nn.Linear(32, 1))
        return model

    def get_empty_state(self):
        return agent.Agent.get_empty_state(self)

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))
   
    '''def remember(self, state, action, reward, new_state, done): 
        self.addToMemory(state, action, reward, new_state, done)
        actor_loss = 0
        critic_loss = 0
        if len(self.memory) < 2*self.batch_size:
            return actor_loss, critic_loss
        mini_batch = self.sample()

        X_train, Y_train = self.calculateTargetActor(mini_batch)
        actor_loss = self.actorModel.train_on_batch(X_train, Y_train)
        Z_train, K_train = self.calculateTargetCritic(mini_batch)
        critic_loss = self.criticModel.train_on_batch(Z_train, K_train)
        self.updateActorNetwork()
        self.updateCriticNetwork()
        return actor_loss, critic_loss'''

    def remember(self, state, action, reward, new_state, done): 
        pass

    def updateNetworks(self, mini_batch):
        X_train, Y_train = self.calculateTargetActor(mini_batch)
        self.actorModel.train_on_batch(X_train, Y_train)
        self.calculateTargetCritic(mini_batch)
        Z_train, K_train = self.calculateTargetCritic(mini_batch)
        self.criticModel.train_on_batch(Z_train, K_train)
        self.updateActorNetwork()
        self.updateCriticNetwork()

    def predict(self, state, isTarget):
        shape = (1,) + self.state_size
        state = np.reshape(state, shape)
        if isTarget:
            value = self.actorTarget.predict([state, self.allMask])
            policy = self.criticTarget.predict([state, self.allMask])
        else:
            value = self.actorModel.predict([state, self.allMask])
            policy = self.criticModel.predict([state, self.allMask])
        return value, policy

    def updateActorNetwork(self):
        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.actorTarget.set_weights(self.actorModel.get_weights())
            print("target actor updated")
        self.total_steps += 1

    def updateCriticNetwork(self):
        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.criticTarget.set_weights(self.criticModel.get_weights())
            print("target critic updated")
        self.total_steps += 1

    def updateCritic(self, advantages):
        critic_optim = Adam(self.parameters(), lr=self.policy_lr)
        loss = 0.5 * (advantages ** 2).mean()
        critic_optim.zero_grad()
        loss.backward()
        critic_optim.step()
        return loss

    def calculateTargetActor(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        next_states = np.zeros((self.batch_size,) + self.state_size)

        for index_rep, transition in enumerate(mini_batch):
            X_train[0][index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        qnext = self.actorTarget.predict([next_states, self.allBatchMask])
        qnext = np.amax(qnext, 1)

        for index_rep, transition in enumerate(mini_batch):
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma
        return X_train, Y_train
    
    def calculateTargetCritic(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        next_states = np.zeros((self.batch_size,) + self.state_size)

        for index_rep, transition in enumerate(mini_batch):
            X_train[0][index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        qnext = self.criticTarget.predict([next_states, self.allBatchMask])
        qnext = np.amax(qnext, 1)

        for index_rep, transition in enumerate(mini_batch):
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma
        return X_train, Y_train


    def update(self):
        pass

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros((vector_length))
        if hot_index != -1:
            output[hot_index] = 1
        return output

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass