import numpy as np
import random

from Agents.Collections import TransitionFrame
from collections import deque

class ReplayBuffer:
    """
    An Experience Replay Buffer for looking back and resampling transitions.
    """
    def __init__(self, learner, max_length, empty_trans, history_length: int = 1):
        """
        Constructor method
        :param learner: the agent using this buffer
        :type learner: Agent
        :param max_length: the max length of this buffer
        :type max_length: int
        :param empty_trans: the empty transition to pad results with
        :type empty_trans: TransitionFrame
        :param history_length: the length of the history
        :type history_length: int
        """
        self._cur_idx = 0
        self.empty_trans = empty_trans
        self.history_length = history_length
        self.learner = learner
        self.max_length = max_length
        self._size = 0
        self._transitions = np.empty(max_length, dtype = object)
    
    def __len__(self):
        """
        Returns the length of this replay buffer.
        :return: the length of the buffer
        :rtype: int
        """
        return self._size
    
    def append_frame(self, transition_frame):
        """
        Appends a given framed to the buffer.
        :param transition_frame: the transition frame to append to the end
        of this buffer
        :type transition_frame: TransitionFrame
        """
        # Add the transition to the buffer.
        self._transitions[self._cur_idx] = transition_frame
        
        # Increment the current index.
        self._cur_idx = (self._cur_idx + 1) % self.max_length
        
        # Increment the size if the size is less than the max_length.
        if (self._size < self.max_length):
            self._size += 1
    
    def get_recent_action(self):
        """
        Get the last n actions where n is equal to the history length of the
        buffer.
        :return: the recent actions
        :rtype: list
        """
        # Empty deque to prepend actions to.
        result = deque(maxlen = self.history_length)
        
        # Get the latest action until the history length or until the beginning
        # of the buffer is reached.
        for i in range((self._cur_idx - 1), ((self._cur_idx - 1) - self.history_length), -1):
            if (i < 0 and not self.is_full()):
                break
            result.appendleft(self._transitions[i % self.max_length].action)
        
        # Prepend -1s until the length of the deque equals the history
        # length.
        while (len(result) < self.history_length):
            result.appendleft(-1)
        
        # Return the recent actions as a list.
        return list(result)
    
    def get_recent_state(self):
        """
        Get the last n states where n is equal to the history length of the
        buffer.
        :return: the recent states
        :rtype: list
        """
        # Empty deque to prepend states to.
        result = deque(maxlen = self.history_length)
        
        # Get the latest states until the history length or until the beginning
        # of the buffer is reached.
        for i in range((self._cur_idx - 1), ((self._cur_idx - 1) - self.history_length), -1):
            if (i < 0 and not self.is_full()):
                break
            result.appendleft(self._transitions[i % self.max_length].state)
        
        # Prepend empty states until the length of the deque equals the
        # history length.
        empty_state = self.empty_trans.state
        while (len(result) < self.history_length):
            result.appendleft(empty_state)
        
        # Return the recent states as a list.
        return list(result)
    
    def get_transitions(self, start):
        """
        Gets a list of transition from the given index. The length of the
        the list will be equal to the history length of the buffer.
        :param start: is the start index to get transtions from.
        :type start: int
        :return: the padded transitions
        :rtype: list
        """
        # Check if the index within bounds.
        if (start < 0 or start >= len(self)):
            raise ValueError("Start index is out of bounds.")
        
        # If the history length is equal to 1, just return the transition
        # at the given index.
        if (self.history_length == 1):
            return self._transitions[start]
            
        # Empty list to store the transitions.
        results = []
        # Iterate through the buffer, adding transitions to the list.
        for i in range(start, start + self.history_length):
            results.append(self._transitions[i])
            if self._transitions[i].is_done or i == (self._cur_idx - 1):
                break
                
        # Pad and return the transitions.
        return self.pad(results)
    
    def is_empty(self):
        """
        Checks whether this replay buffer is empty.
        :return: true if this replay buffer is empty, false otherwise
        :rtype: int
        """
        return self._size == 0
    
    def is_full(self):
        """
        Checks whether this replay buffer has reached the max length.
        :return: true if this replay buffer is full, false otherwise
        :rtype: int
        """
        return self._size == self.max_length
    
    def pad(self, transitions):
        """
        Adds padding to the beginning of the given list of transitions.
        :param transitions: the list of transitions to pad
        :type transitions: list
        :return: the padded transitions
        :rtype: list
        """
        return [self.empty_trans for _ in
               range(self.history_length - len(transitions))] + transitions
    
    def peak_frame(self):
        """
        Returns the last frame if the buffer is non-empty and an empty
        transition frame otherwise.
        :return: the last frame added to the buffer
        :rtype: TransitionFrame
        """
        if (self.is_empty()):
            return self.empty_trans
        return self._transitions[self._cur_idx - 1]
    
    def sample(self, batch_size):
        """
        Gets a number of samples equal equal to the batch size, each length
        equal to the history length of this buffer.
        :return: the length of the buffer
        :rtype: int
        """
        result = []
        for i in random.sample(range(len(self)), batch_size):
            result.append(self.get_transitions(i))
        return result
