import torch.nn as nn
import torch.nn.functional as F

from Agents import agent, modelFreeAgent
from Agents.deepQ import DeepQ
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, multiply, Lambda


class Actor(DeepQ):
  def __init__(self, policy_lr):
    super()._init_()
    self.policy_model = self.policy_network()
    self.policy_lr = policy_lr
    self.optim = tf.keras.optimizers.Adam(self.policy_lr)

  def policy_network(self):
    model = tf.keras.Sequential([
            Input((self.state_size,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='softmax')
    ])
    kl = tf.keras.losses.KLDivergence()
    model.compile(loss=kl, optimizer=Adam(lr = self.policy_lr))
    return model

class Critic(DeepQ):
  def __init__(self, value_lr):
    super().__init__()
    self.value_lr = value_lr
    self.value_model = self.value_network()
    self.optim = tf.keras.optimizers.Adam(self.value_lr)

  def value_network(self):
    model = tf.keras.Sequential([
            Input((self.state_size,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(lr = self.value_lr))
    return model
