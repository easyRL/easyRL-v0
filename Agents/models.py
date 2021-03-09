import torch.nn as nn
import torch.nn.functional as F

from Agents import agent, modelFreeAgent
from Agents.deepQ import DeepQ
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, multiply, Lambda
import sys

class Actor(DeepQ):
  def __init__(self, state_size, action_size, policy_lr):
    self.state_size = state_size
    self.action_size = action_size
    self.policy_lr = policy_lr
    self.policy_model = self.policy_network()
    #self.optim = tf.keras.optimizers.Adam(self.policy_lr)

  def policy_network(self):
    try:
      # inputA = Input(shape=self.state_size)
      # inputA = Flatten()(inputA)
      print("State size: " + str(self.state_size) + "\n")
      model = tf.keras.Sequential([
              Dense(32, activation='relu', input_shape=(self.state_size)),
              Dense(16, activation='relu', input_shape=(self.state_size)),
              Dense(self.action_size, activation='softmax', input_shape=(self.state_size))
      ])
      kl = tf.keras.losses.KLDivergence()
      model.compile(loss=kl, optimizer=Adam(lr = self.policy_lr))
      return model
    except:
      print("\n\n\n")
      print(sys.exc_info())
      sys.exit()

class Critic(DeepQ):
  def __init__(self, state_size, action_size, value_lr):
    self.state_size = state_size
    self.action_size = action_size
    self.value_lr = value_lr
    self.value_model = self.value_network()

  def value_network(self):
    # inputA = Input(shape=self.state_size)
    # inputA = Flatten()(inputA)
    model = tf.keras.Sequential([
            Dense(32, activation='relu', input_shape=(self.state_size)),
            Dense(16, activation='relu', input_shape=(self.state_size)),
            Dense(16, activation='relu', input_shape=(self.state_size)),
            Dense(1, activation='linear', input_shape=(self.state_size))
    ])
    model.compile(loss='mse', optimizer=Adam(lr = self.value_lr))
    return model
