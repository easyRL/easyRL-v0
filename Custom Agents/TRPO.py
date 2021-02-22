from Agents import ppo
import numpy as np
import tensorflow as tf
#from tensorflow.linalg.experimental import conjugate_gradient
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Flatten, TimeDistributed, LSTM, multiply
from tensorflow.keras import utils
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam

class TRPO(ppo):
    displayName = 'TRPO Agent'
    #Invoke constructor
    def __init__(self, *args):
        paramLen = len(super().newParameters)
        super().__init__(*args[:-paramLen])
        self.actorModel = ppo.buildQNetwork()
        self.actorModel.compile(loss=tf.keras.losses.KLDivergence(), optimizer=Adam(lr=self.value_lr, clipvalue=1))
        self.criticModel = ppo.buildQNetwork()
        self.criticModel.compile(loss=tf.keras.losses.KLDivergence(), optimizer=Adam(lr=self.policy_lr, clipvalue=1))

    def optimize(self, action, state, policy, parameters, newParameters):
        for critic in range(10):
            for actor in range(10):
                self.update()
                # compute advantage
            #optimize surrogate loss using recurrent neural network 
            parameters = newParameters

    def calculateTargetValues(self, mini_batch):
        pass

    def update(self):
        super().update()

    def remember(self, state, action, reward, new_state, done): 
        pass

    def reset(self):
        pass

    def __deepcopy__(self, memodict={}):
        pass