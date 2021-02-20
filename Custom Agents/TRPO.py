from Agents import agent, actorCriticNative
from Agents.deepQ import DeepQ
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply
from tensorflow.python.keras import utils
from tensorflow.python.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam

class TRPO(actorCriticNative, DeepQ):
    displayName = 'TRPO Agent'
    self.model = buildQNetwork()
    #Invoke constructor
    def __init__(self, *args):
        self.parameters = super.parameters
        self.newParameters = super.newParameters
        self.action_size = agent.action_size
        self.state_size = agent.state_size
        actorCriticNative.__init__(*args)
        self.rewards = []

    def probabilityRatio(self, action, state, policy, parameters, newParameters):
        self.ratio = 

    def PPO(self, action, state, policy, parameters, newParameters):
        for critic in range(1000):
            for actor in range(100):
                self.update()
                # compute advantage
            #optimize surrogate loss using recurrent neural network 
            parameters = newParameters

    def buildQNetwork(self):
        input_size = self.state_size
        output_size = self.action_size
        inputA = Input(shape=input_shape)
        inputB = Input(shape=(self.action_size,))

        if len(self.state_size) == 1:
            x = TimeDistributed(Dense(10, input_shape=input_shape, activation='relu'))(inputA)
        else:
            x = TimeDistributed(Conv2D(16, 8, strides=4, activation='relu'))(inputA)
            x = TimeDistributed(Conv2D(32, 4, strides=2, activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(256)(x)
        x = Dense(10, activation='relu')(x)  # fully connected
        x = Dense(10, activation='relu')(x)
        x = Dense(self.action_size)(x)
        outputs = multiply([x, inputB])
        kl = tf.keras.losses.KLDivergence()
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        model.compile(loss=kl, optimizer=Adam(lr=0.0001, clipvalue=1))
        return model

    def calculateTargetValues(self, mini_batch):
        super.calculateTargetValues(mini_batch)

    def update(self):
        super.update()

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass