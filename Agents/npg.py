import copy
import joblib
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agents import policyIteration
from Agents.Policy import approximator, policy
from collections.abc import Iterable
from torch.autograd import Variable
from torch.distributions import Categorical, kl
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

class NPG(policyIteration.PolicyIteration):
    """
    Natural Policy Gradient. A policy iteration agent that updates the
    policy using the policy gradient.
    
    Adapted from 'https://github.com/zafarali/policy-gradient-methods/
    blob/master/pg_methods/experimental/npg/npg_algorithm.py'.
    """
    displayName = 'NPG'
    newParameters = [policyIteration.PolicyIteration.Parameter('Delta', 0, 0.05, 0.0001, 0.001, True, True, "The normalized step size for computing the learning rate.")]
    parameters = policyIteration.PolicyIteration.parameters + newParameters
    
    def __init__(self, *args):
        """
        Constructor for Natural Policy Gradient agent.
        """
        paramLen = len(NPG.newParameters)
        super().__init__(*args[:-paramLen])
        self.delta = float(args[-paramLen])
        
        '''
        Define the policy.
        '''
        # Create a deep learning approximator. MUST USE PYTORCH!
        approx = approximator.DeepApproximator(self.state_size, self.action_size, [16], library = 'torch')
        # Create a categorical policy with a deep approximator for this agent.
        self._policy = policy.CategoricalPolicy(approx)
        
        # Baseline approximator to approximate the value function. MUST USE PYTORCH!
        self._value_fn = approximator.DeepApproximator(self.state_size, 1, [16], library = 'torch')
        
    def choose_action(self, state):
        """
        Chooses an action given the state and, if given, a policy. The
        policy p parameter is optional. If p is None, then the current
        policy of the agent will be used. Otherwise, the given policy p is
        used.
        :param state: is the current state of the environment
        :return: the chosen action
        :rtype: int
        """
        # Choose and return an action using the current policy.
        return self._policy.choose_action(np.asarray(state))
        
    def update(self, trajectory: Iterable):
        """
        Updates the current policy given a the trajectory of the policy.
        :param trajectory: a list of transition frames from the episode.
        This represents the trajectory of the episode.
        :type trajectory: Iterable
        :return: the loss from this update
        :rtype: float:
        """
        if (not isinstance(trajectory, Iterable)):
            raise ValueError("trajectory must be an Iterable.")
        # Consolidate the state in the trajectory into an array.
        states = np.array([np.asarray(transition.state) for transition in trajectory])
        
        '''
        Compute the loss as the log-likelihood of the returns.
        '''
        # Calculate the returns.
        returns = self._calculate_returns(trajectory)
        # Calculate the values using the baseline approximator.
        values = torch.Tensor([self._value_fn(state)[0] for state in states])
        # Calculate the advantage using the returns and the values.
        advantages = returns - values
        # Compute the loss of the trajectory.
        logits = torch.stack([self._policy.logit(np.asarray(transition.state), transition.action, detach = False) for transition in trajectory]).view(-1)
        loss = (-logits * advantages).mean()
        
        '''
        Compute the gradient and the natural policy gradient.
        '''
        # Calculate the gradient of the log likelihood loss.
        gradient = self._compute_gradient(loss)
        gradient = parameters_to_vector(gradient).detach().numpy()
        
        # Calculate the natural policy gradient.
        npg = self._compute_npg(gradient, states)
        
        '''
        Update the policy and the baseline.
        '''
        # The learning rate to apply for the update.
        alpha = np.sqrt(np.abs(self.delta / (np.dot(gradient.T, npg.detach().numpy()) + 1e-20)))
        # The amount to change the parameters by.
        update = alpha * npg
        # Calculate and set the new parameters of the policy.
        new_params = parameters_to_vector(self._policy.get_params(False)) - update
        self._policy.set_params(new_params.detach().numpy())
        
        # Update baseline approximator using the cumulative returns.
        self._value_fn.update(states, returns.detach().numpy().reshape(-1, 1))
        
        # Return the loss from the update.
        return loss.item()
    
    def _calculate_returns(self, trajectory: Iterable):
        """
        Calculate the discounted cumualtive rewards of the trajectory.
        :param trajectory: a list of transition for a given trajectory
        :type trajectory: Iterable
        :return: the discounted cumulative returns for each transition
        :rtype: torch.Tensor
        """
        if (not isinstance(trajectory, Iterable)):
            raise ValueError("trajectory must be an Iterable.")
            
        # Calculate the discounted cumulative rewards for each transition
        # in the trajectory.
        returns = torch.zeros(len(trajectory))
        returns[-1] = trajectory[-1].reward
        for t in reversed(range(len(trajectory) - 1)):
            returns[t] = (1 - trajectory[t].is_done) * trajectory[t].reward + self.gamma * returns[t+1]
        
        # Return the discounted cumulative rewards.
        return returns
    
    def _compute_gradient(self, output: torch.Tensor):
        """
        Computes the gradient of the given output, using the parameters of
        the policy as the inputs.
        :param output: is the output to compute the gradient of
        :type output: torch.Tensor
        :return: the calculated gradient
        :rtype: torch.Tensor
        """
        if (not isinstance(output, torch.Tensor) or output.requires_grad == False):
            raise ValueError("The output must be a torch.Tensor with a grad_fn.")
        
        # Zero out the gradient before computing.
        self._policy.zero_grad()
        # Compute and return the gradient of the loss using autograd.
        gradient = torch.autograd.grad(output, self._policy.get_params(False), retain_graph=True, create_graph=True)
        return gradient
  
    def _compute_hvp(self, gradient: np.ndarray, states: np.ndarray, regularization: float = 1e-9):
        """
        Computes the Hessian Vector Product (HVP) of the gradient
        :param gradient: is the gradient to compute the HVP of
        :type gradient: numpy.ndarray
        :param states: is the states of the trajectory from which the
        gradient was calculated
        :type states: numpy.ndarray
        :param regularization: amount of regularization to apply
        :type regularization: float
        :return: the Hessain Vector Product of the gradient
        :rtype: torch.Tensor
        """
        if (not isinstance(gradient, np.ndarray)):
            raise ValueError("gradient must be a numpy array.")
        if (not isinstance(states, np.ndarray) or states.shape[1:] != self.state_size):
            raise ValueError("states must be a numpy array with each state having the shape {}.".format(self.state_size))
        if (not isinstance(regularization, float)):
            raise ValueError("regularization must be a float.")
        
        # Convert the gradient into a Tensor.
        gradient = torch.from_numpy(gradient).float()
        
        # Zero out the gradient of the current policy.
        self._policy.zero_grad()
        
        # Calculate the KL divergence of the old and the new policy distributions.
        old_policy = copy.deepcopy(self._policy).get_distribution(states)
        new_policy = self._policy.get_distribution(states, detach = False)
        mean_kl = torch.mean(kl.kl_divergence(new_policy, old_policy))
        
        # Calculate the gradient of the KL divergence.
        kl_grad = torch.autograd.grad(mean_kl, self._policy.get_params(False), create_graph=True)
        
        # Calculate the gradient of the KL gradient to the HVP.
        h = torch.sum(parameters_to_vector(kl_grad) * gradient)
        hvp = torch.autograd.grad(h, self._policy.get_params(False))
        # Flatten the HVP into a one-dimensional tensor.
        hvp_flat = np.concatenate([g.contiguous().view(-1).numpy() for g in hvp])
        hvp_flat = torch.from_numpy(hvp_flat)
        
        # Return the flatten HVP plus the regularized gradient.
        return hvp_flat + regularization * gradient

    def _compute_npg(self, gradient: np.ndarray, states: np.ndarray, iters: int = 1, max_residual: float = 1e-10):
        """
        Computes the Natural Policy Gradient (NPG) of the policy and the 
        given gradient.
        
        Adapted from 'https://github.com/zafarali/policy-gradient-methods/
        blob/f0d83a80ddc772dcad0c851aac9bfd41d436c274/pg_methods/
        conjugate_gradient.py'.
        
        :param gradient: the gradient to compute the NPG of
        :type gradient: numpy.ndarray
        :param states: the states from the trajectory that generate the
        gradient
        :type states: numpy.ndarray
        :param iters: the number of iteration of conjugation to perform
        :type iters: int
        :param max_residual: the maximum residual allowed during conjugation
        :type max_residual: float
        :return: the Natural Policy Gradient of the policy and the given
        gradient
        :rtype: Torch.tensor
        """
        if (not isinstance(gradient, np.ndarray)):
            raise ValueError("gradient must be a numpy array.")
        if (not isinstance(states, np.ndarray) or states.shape[1:] != self.state_size):
            raise ValueError("states must be a numpy array with each state having the shape {}.".format(self.state_size))
        if (not isinstance(iters, int) or iters < 1):
            raise ValueError("iters must be a positive integer.")
        if (not isinstance(max_residual, float)):
            raise ValueError("regularization must be a float.")
        
        p = gradient.copy()
        r = gradient.copy()
        x = np.zeros_like(gradient)
        rdotr = r.dot(r)

        for i in range(iters):
            z = self._compute_hvp(p, states)
            v = rdotr / p.dot(z)
            x += v*p
            r -= (v*z).detach().numpy()
            newrdotr = r.dot(r)
            mu = newrdotr/rdotr
            p = r + mu*p
            
            rdotr = newrdotr
            if rdotr < max_residual:
                break

        return torch.from_numpy(x)

    def save(self, filename):
        mem = self._policy.get_params()
        joblib.dump((NPG.displayName, mem), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != NPG.displayName:
            print('load failed')
        else:
            self._policy.set_params(mem)

    def memsave(self):
        return self._policy.get_params()

    def memload(self, mem):
        self._policy.set_params(mem)
