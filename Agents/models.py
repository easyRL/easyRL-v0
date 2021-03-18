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
    '''try:
      # inputA = Input(shape=self.state_size)
      # inputA = Flatten()(inputA)
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
      sys.exit()'''
    from tensorflow.python.keras.optimizer_v2.adam import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input, Flatten, multiply
    inputA = Input(shape=self.state_size)
    inputB = Input(shape=(self.action_size,))
    x = Flatten()(inputA)
    x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
    x = Dense(24, activation='relu')(x)
    outputs = Dense(self.action_size, activation='softmax')(x)
    model = Model(inputs=[inputA, inputB], outputs=outputs)
    kl = tf.keras.losses.KLDivergence()
    model.compile(loss=kl, optimizer=Adam(lr=self.policy_lr))
    return model

class Critic(DeepQ):
  def __init__(self, state_size, action_size, value_lr):
    self.state_size = state_size
    self.action_size = action_size
    self.value_lr = value_lr
    self.value_model = self.value_network()

  def value_network(self):
    # inputA = Input(shape=self.state_size)
    # inputA = Flatten()(inputA)
    '''model = tf.keras.Sequential([
            Dense(32, activation='relu', input_shape=(self.state_size)),
            Dense(16, activation='relu', input_shape=(self.state_size)),
            Dense(16, activation='relu', input_shape=(self.state_size)),
            Dense(1, activation='linear', input_shape=(self.state_size))
    ])
    model.compile(loss='mse', optimizer=Adam(lr = self.value_lr))'''
    from tensorflow.python.keras.optimizer_v2.adam import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input, Flatten, multiply

    inputA = Input(shape=self.state_size)
    inputB = Input(shape=(self.action_size,))
    x = Flatten()(inputA)
    x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
    x = Dense(24, activation='relu')(x)
    x = Dense(self.action_size, activation='linear')(x)
    outputs = multiply([x, inputB])
    model = Model(inputs=[inputA, inputB], outputs=outputs)
    model.compile(loss='mse', optimizer=Adam(lr=self.value_lr))
    return model
