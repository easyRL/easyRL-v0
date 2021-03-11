import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod
from collections.abc import Iterable

class Approximator(ABC):
    """
    Interface for function approximator. A function approximator is a
    class that approximates the value of action given a state.
    """
    def __init__(self, state_size: tuple, action_size: int):
        """
        Constructor for Approximator.
        :param state_size: is the shape and size of the state
        :type state_size: tuple
        :param action_size: is the size of the action space
        :type action_size: int
        """
        if (not isinstance(state_size, tuple)):
            raise ValueError("state_size must be a tuple of positive integers.")
        if (not isinstance(action_size, int) or action_size < 1):
            raise ValueError("action_size must be a positive integer.")
        self.state_size = state_size
        self.action_size = action_size
    
    @abstractmethod
    def __call__(self, state: np.ndarray):
        """
        Approximates the value of each action in the action space given
        the state.
        :param state: the current state of the environment
        :type state: numpy.ndarray
        :return: the approximate values outputted by the approximator.
        """
        pass
    
    @abstractmethod
    def count_params(self):
        """
        Counts the number of parameters for this approximator.
        :return: the number of parameters for this approximator
        :rtype: int
        """
        pass
    
    @abstractmethod
    def get_params(self, flatten: bool = True):
        """
        Gets the parameters used by this approximator. The parameters
        for this model is the weights and bias of each layer. The
        parameters are returned as a one-dimensional numpy array if flatten
        is true which is by default, otherwise the parameters are return
        in the format of the model being used.
        :param flatten: whether to flatten the parameters to a
        one-dimensional array or not
        :type param: bool
        :return: the parameters used by this approximator
        """
    
    @abstractmethod
    def set_params(self, params: np.ndarray):
        """
        Set the parameters of this model to the ones given in the
        parameters as a numpy array. The length of the array must equal
        the number of parameters used by this model. The parameters must
        be flattened into a one-dimensional numpy array.
        :param params: A numpy array of the parameters to set this
        approximator to.
        :type params: numpy.ndarray
        """
        pass
    
    @abstractmethod
    def update(self, states: np.ndarray, targets: np.ndarray):
        """
        Updates the approximator given a batch of states as the input and
        their corresponding target values.
        :param states: an array of multiple states that are the input into
        approximator to approximator the value of.
        :type states: numpy.ndarray
        :param targets: the target values the approximator should calculate.
        :type targets: numpy.ndarray
        :return: the loss from training.
        :rtype: float
        """
        pass
    
    @abstractmethod
    def zero_grad(self):
        """
        Zeros out the gradient of the model.
        """
        pass

class DeepApproximator(Approximator):
    """
    Artificial Neural Network implementation of a function approximator.
    Can use either keras or Pytorch to construct the sequential model.
    """
    # List of available machine learning libraries.
    libraries = ['torch']
    '''
    No longer allows the use of Keras as the library as it needs more
    work to be implemented by policy.py and consequently an agent that use
    Deep Approximator since __call__ will return different types of tensors.
    
    The methods that need to be tweaked in DeepApproximator to fully
    support keras are __call__ (it needs to output a TensorFlow tensor),
    update, and zero_grad.
    
    Comment back in when these are updated to support keras and policy.py
    and any agent using DeepApproximator can use either libraries since the
    tensors are different.
    '''
    #libraries = ['keras', 'torch']
    
    def __init__(self, state_size: tuple, action_size: int, hidden_sizes: Iterable = [], library: str = 'torch'):
        """
        Constructor for DeepApproximator.
        :param state_size: is the shape and size of the state
        :type state_size: tuple
        :param action_size: is the size of the action space
        :type action_size: int
        :param hidden_sizes: An Iterable object contain the lengths of
        hidden layers to add to the architecture
        :type hidden_sizes: Iterable
        """
        if (not isinstance(hidden_sizes, Iterable)):
            raise ValueError("hidden_sizes must be an Iterable object.")
        if (library not in DeepApproximator.libraries):
            raise ValueError("{} is not a valid machine learning library. Use one from the following list: {}".format(library, DeepApproximator.libraries))
        
        # Call super constructor.
        super(DeepApproximator, self).__init__(state_size, action_size)
        
        self.hidden_sizes = hidden_sizes
        self.library = library
        
        # Construct the model of the approximator function.
        self._model = None
        self._optimizer = None
        if (self.library == 'keras'):
            self._model = self._build_keras_network()
        if (self.library == 'torch'):
            self._model = self._build_torch_network()
            self._optimizer = optim.Adam(self._model.parameters(), lr = 0.001)
    
    def __call__(self, state: np.ndarray):
        """
        Approximates the value of each action in the action space given
        the state.
        :param state: the current state of the environment
        :type state: numpy.ndarray
        :return: the approximate values outputted by the approximator.
        """
        if (not isinstance(state, np.ndarray) or state.shape != self.state_size):
            raise ValueError("state must be a numpy.ndarray with shape {}".format(self.state_size))
        
        # Initialize the values as an empty array.
        values = np.empty(0)
        
        # Approximate the values.
        if (self.library == 'keras'):
            # Reshape the state to input into the model.
            state = np.reshape(state, (1,) + self.state_size)
            # Approximate the values and reshape the values.
            values = self._model.predict(state)
            values = np.reshape(values, -1)
        elif (self.library == 'torch'):
            # Convert the state into a tensor.
            state = torch.from_numpy(state).float()
            # Approximate the values and convert the results to an array.
            values = self._model(state)
        
        # Return the values.
        return values
        
    def count_params(self):
        """
        Counts the number of parameters for this approximator.
        :return: the number of parameters for this approximator
        :rtype: int
        """
        if (self.library == 'keras'):
            return self._model.count_params()
        elif (self.library == 'torch'):
            return sum(param.numel() for param in self._model.parameters())
        
    def get_params(self, flatten: bool = True):
        """
        Gets the parameters used by this approximator. The parameters
        for this model is the weights and bias of each layer. The
        parameters are returned as a one-dimensional numpy array if flatten
        is true which is by default, otherwise the parameters are return
        in the format of the model being used.
        :param flatten: whether to flatten the parameters to a
        one-dimensional array or not
        :type param: bool
        :return: the parameters used by this approximator
        """
        # Get the parameters in the raw form
        params = None
        if (self.library == 'keras'):
            params = self._model.get_weights()
        elif (self.library == 'torch'):
            params = self._model.parameters()
        
        '''
        If flatten is true, flatten the parameters to a one-dimensional
        numpy array.
        '''
        if (flatten):
            # Empty numpy array to append the parameters to.
            flat_params = np.empty(0)
            
            # Get the parameters from the model and append them to the array.
            if (self.library == 'keras'):
                for layer_params in params:
                    flat_params = np.append(flat_params, layer_params)
            elif (self.library == 'torch'):
                for layer_params in params:
                    flat_params = np.append(flat_params, layer_params.detach().numpy())
            
            # Set the parameters to the flatten representation.
            params = flat_params
        
        # Return the parameters.
        return params
        
    def set_params(self, params: np.ndarray):
        """
        Set the parameters of this model to the ones given in the
        parameters as a numpy array. The length of the array must equal
        the number of parameters used by this model. The parameters must
        be flattened into a one-dimensional numpy array.
        :param params: A numpy array of the parameters to set this
        approximator to.
        :type params: numpy.ndarray
        """
        if (not isinstance(params, np.ndarray)):
            raise ValueError("params must be an numpy array.")
        if (params.dtype is np.float64):
            raise ValueError("params must have float64 as its dtype.")
        if (len(params) != self.count_params()):
            raise ValueError("params must have length equal to the number of parameter used by this approximator.")
        
        # Get the indices for the weights and bias of each layer.
        layer_idxes = self._layer_idxes()
        
        # Set the weights and bias for each layer.
        if (self.library == 'keras'):
            # Get the layers.
            layers = self._model.layers[1:]
            for layer_idx, layer in zip(layer_idxes, layers):
                # Get the weights and bias for this layer.
                w = np.reshape(params[layer_idx[0]: layer_idx[1]], layer.get_weights()[0].shape)
                b = np.reshape(params[layer_idx[1]: layer_idx[2]], layer.get_weights()[1].shape)
                # Set the weights and bias for this layer to what was given.
                layer.set_weights([w, b])
        elif (self.library == 'torch'):
            # Get the layers.
            layers = [l for l in self._model.modules() if isinstance(l, nn.Linear)]
            for layer_idx, layer in zip(layer_idxes, layers):
                # Get the weights and bias for this layer.
                w = np.reshape(params[layer_idx[0]: layer_idx[1]], layer.weight.data.shape)
                b = np.reshape(params[layer_idx[1]: layer_idx[2]], layer.bias.data.shape)
                # Convert the weights and bias to tensors.
                w = torch.from_numpy(w)
                b = torch.from_numpy(b)
                # Set the weights and bias for this layer to what was given.
                layer.weight.data.copy_(w.view_as(layer.weight.data))
                layer.bias.data.copy_(b.view_as(layer.bias.data))
    
    def update(self, states: np.ndarray, targets: np.ndarray):
        """
        Updates the approximator given a batch of states as the input and
        their corresponding target values.
        :param states: an array of multiple states that are the input into
        approximator to approximator the value of.
        :type states: numpy.ndarray
        :param targets: the target values the approximator should calculate.
        :type targets: numpy.ndarray
        :return: the loss from training.
        :rtype: float
        """
        if (not isinstance(states, np.ndarray) or states.shape[1:] != self.state_size):
            raise ValueError("states must be a numpy array with each state having the shape {}.".format(self.state_size))
        if (not isinstance(targets, np.ndarray) or targets.shape[1:] != (self.action_size,)):
            raise ValueError("targets must be a numpy array with each target having the shape ({},).".format(self.action_size))
        
        # Zero out the gradients.
        self._optimizer.zero_grad()
        
        # Approximate the values of each state using the model.
        approx_values = torch.zeros(len(targets), 1)
        for idx, state in enumerate(states):
            # Convert the state to a tensor.
            state = torch.from_numpy(state).float()
            # Approximate the values of the state.
            approx_values[idx][0] = self._model(state)
        
        # Calculate the loss as the MSE.
        targets = torch.from_numpy(targets)
        loss_fn = nn.MSELoss()
        loss = loss_fn(approx_values, targets)
        
        # Backpropagate the loss.
        loss.backward()
        self._optimizer.step()
        
        # Return the loss.
        return loss.item()
    
    def zero_grad(self):
        """
        Zeros out the gradient of the model.
        """
        self._model.zero_grad()
    
    def _build_keras_network(self):
        """
        Constructs and returns a sequential model using Keras. The model
        architecture is an Artificial Neural Network that will flatten the
        input. Each hidden layer uses ReLU activation.
        :return: sequential model to use as the function approximator
        :rtype: torch.nn.Sequential
        """        
        # Import the necessary packages for keras.
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Input, Flatten
        
        # Sequential model to build the architecture for.
        model = Sequential()
        
        '''
        Construct Model Architecture.
        '''  
        # Input and flatten layers to accept and flatten the state as
        # input.
        model.add(Input(shape = self.state_size))
        model.add(Flatten())
        # Create and add n hidden layers sequentially to the architecture,
        # where n is the length of hidden_sizes.
        for size in self.hidden_sizes:
            model.add(Dense(size, activation= 'relu'))
        # Create the output layer.
        model.add(Dense(self.action_size, activation= 'linear'))
        
        # Compile and return the model.
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    
    def _build_torch_network(self):
        """
        Constructs and returns a sequential model using PyTorch. The model
        architecture is an Artificial Neural Network that will flatten the
        input. Each hidden layer uses ReLU activation.
        :return: sequential model to use as the function approximator
        :rtype: torch.nn.Sequential
        """
        
        # Ordered list for the layers and other modules.
        layers = []
        
        '''
        Construct Model Architecture.
        '''
        # Flattening layer to flatten the input if necessary.
        layers.append(nn.Flatten(0, -1))
        
        if (len(self.hidden_sizes) == 0):
            # No hidden layers, connect input layer to output layer directly.
            layers.append(nn.Linear(np.prod(self.state_size), self.action_size))
        else:
            # Construct architecture with n hidden layers, where n is the
            # length of hidden_sizes.
            # Create input layer and connect to first hidden layer.
            layers.append(nn.Linear(np.prod(self.state_size), self.hidden_sizes[0]))
            layers.append(nn.ReLU())
            
            # Create n-1 additional hidden layers and connect them sequentially.
            for i in range(len(self.hidden_sizes) - 1):
                layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
                layers.append(nn.ReLU())
            
            # Create output layer and connect last hidden layer to it.
            layers.append(nn.Linear(self.hidden_sizes[-1], self.action_size))
                
        # Compile and return the sequential model.
        return nn.Sequential(*layers)
    
    def _layer_idxes(self):
        """
        Calculates the indices tuples that point to where the weights and
        bias are located in the array of parameters for each layer. The
        i-th tuple represents the indices (weights_start, weights_end/
        bias_start, bias_end) for the i-th layer.
        :return: list of tuples that describe how to parse the weights and
        bias of each layer from the parameters.
        :rtype: list
        """
        # Empty list for storing tuples of the indices
        idxes = []
        
        # Calculate the indice tuples
        if (len(self.hidden_sizes) == 0):
            # Indices of weights and bias from input to output layer.
            offset = 0
            fc_length = np.prod(self.state_size) * self.action_size
            idxes.append((offset, offset + fc_length, offset + fc_length + self.action_size))
        else:
            # Indices of weights and bias from input to first hidden layer.
            offset = 0
            fc_length = np.prod(self.state_size) * self.hidden_sizes[0]
            idxes.append((offset, offset + fc_length, offset + fc_length + self.hidden_sizes[0]))
            # Indices of weights and bias from hidden layer i to hidden layer i+1.
            for i in range(len(self.hidden_sizes) - 1):
                offset = idxes[-1][2]
                fc_length = self.hidden_sizes[i] * self.hidden_sizes[i + 1]
                idxes.append((offset, offset + fc_length, offset + fc_length + self.hidden_sizes[i + 1]))
            # Indices of weights and bias from last hidden layer output.
            offset = idxes[-1][2]
            fc_length = self.hidden_sizes[-1] * self.action_size
            idxes.append((offset, offset + fc_length, offset + fc_length + self.action_size))
        
        # Return 
        return idxes
