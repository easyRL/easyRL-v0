import numpy as np
import random

from Agents.Collections.TransitionFrame import TransitionFrame, ActionTransitionFrame
from collections import deque
from collections.abc import Iterable
from copy import deepcopy

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
            transition = self._transitions[i % self.max_length]
            results.append(transition)
            if transition.is_done or i == (self._cur_idx - 1):
                break
                
        # Pad and return the transitions.
        return self._pad(results)
      
    def get_next_transitions(self, start, end):
        result = []
        for i in range(start, end):
            result.append(self._transitions[i])
        return result
    
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

    def _pad(self, transitions):
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
    
    def sample(self, batch_size: int):
        """
        Gets a number of samples equal equal to the batch size, each length
        equal to the history length of this buffer.
        :param batch_size: The number of samples to retrieve
        :type batch_size: int
        :return: the sample indexes and samples in lists
        :rtype: tuple of lists
        """
        if (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("The batch size must be a positive integer.")
        
        result_idxes = []
        result = []
        for i in random.sample(range(len(self)), batch_size):
            result_idxes.append(i)
            result.append(self.get_transitions(i))
        return result_idxes, result
    
    def sum_priority(self):
        """
        Returns the sum of priorities of all transitions in this
        Prioritized Replay Buffer.
        :return: the sum of all priorities
        :rtype: float
        """
        pass
    
    def update_error(self, idx: int, error: float):
        """
        Updates the priority of the transition frame at the given idx given
        the error of the transition.
        :param idx: is the index of the transition to update the priority of
        :type idx: int
        :param error: is error of the transition
        :type error: float
        """
        pass
    
    def update_priority(self, idx: int, priority: float):
        """
        Updates the priority of the transition frame at the given idx directly.
        :param idx: is the index of the transition to update the priority of
        :type idx: int
        :param priority: is priority to give the transition
        :type priority: float
        """
        pass

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    An Experience Replay Buffer for looking back and resampling transitions
    using a prioritized sampling technique based on loss.
    
    Requires the agent to have a compute_loss function. The agent needs to
    compute the loss of a sample and call either update_error or
    update_priority to update the sample's priority.
    """
    def __init__(self, learner, max_length, empty_trans, history_length: int = 1, alpha: float = 0.6):
        super().__init__(learner, max_length, empty_trans, history_length)
        self._alpha = alpha
        self._epsilon = 0.00001
        self._max_priority = 1.0
        '''
        A SumTree that stores the priorities of the transitions. The leaf
        nodes, the second half tree array, are the direct priorities of the
        transitions and the parent nodes contain the sum of the priorities
        of all of their children.
        '''
        self._priority_tree = np.zeros(2 * self.max_length - 1)
    
    def append_frame(self, transition_frame):
        """
        Appends a given framed to the buffer.
        :param transition_frame: the transition frame to append to the end
        of this buffer
        :type transition_frame: TransitionFrame
        """
        # Add the transition_frame to the transitions array.
        super().append_frame(transition_frame)
        
        # Set the priority for this transition as the max priority.
        self.update_priority((self._cur_idx - 1) % self.max_length, self._max_priority)
    
    def sample(self, batch_size: int):
        """
        Gets a number of samples equal equal to the batch size, each length
        equal to the history length of this buffer.
        :param batch_size: The number of samples to retrieve
        :type batch_size: int
        :return: the sample indexes and samples in lists
        :rtype: tuple of lists
        """
        if (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("The batch size must be a positive integer.")

        result_idx = []
        result = []
        segment_length = self.sum_priority() / batch_size
        for i in range(batch_size):
            s = random.uniform(segment_length * i, segment_length * (i + 1))
            sample_idx = self._sample_helper(0, s)
            result_idx.append(sample_idx)
            result.append(self.get_transitions(sample_idx))
        
        return result_idx, result
    
    def sum_priority(self):
        """
        Returns the sum of priorities of all transitions in this
        Prioritized Replay Buffer.
        :return: the sum of all priorities
        :rtype: float
        """
        return self._priority_tree[0]
    
    def update_error(self, idx: int, error: float):
        """
        Updates the priority of the transition frame at the given idx given
        the error of the transition.
        :param idx: is the index of the transition to update the priority of
        :type idx: int
        :param error: is error of the transition
        :type error: float
        """
        if (not isinstance(error, float)):
            raise ValueError("The error should be a float")
            
        # Calculate the priority from the error.
        priority = self._calculate_priority(error)
        # Update the transition with the calculated priority.
        self.update_priority(idx, priority)
    
    def update_priority(self, idx: int, priority: float):
        """
        Updates the priority of the transition frame at the given idx directly.
        :param idx: is the index of the transition to update the priority of
        :type idx: int
        :param priority: is priority to give the transition
        :type priority: float
        """
        if (not isinstance(idx, int) or idx < 0 or idx >= self._size):
            raise ValueError("The index must be a valid index in the range of the buffer.")
        if (not isinstance(priority, float)):
            raise ValueError("The priority must be a float.")
        
        # Update the priority of the given transition and propagate to
        # change to the parent nodes.
        tree_idx = self._to_tree_idx(idx)
        self._propagate(tree_idx, (priority ** self._alpha) - self._priority_tree[tree_idx])
        
        # Update the max priority if necessary.
        self._max_priority = max(self._max_priority, priority)
    
    def _calculate_priority(self, error: float):
        """
        Calculates the priority from the given error.
        :param error: is the error
        :type error: float
        :return: the calculated priority
        :rtype: float
        """
        return (np.abs(error) + self._epsilon) ** self._alpha
    
    def _propagate(self, idx: int, change: float):
        """
        Recursively propogates the change in the priority of a node up the
        SumTree until the change is propagated to the root.
        :param idx: is the index for a node in the array to propagate the
        change to
        :type idx: int
        :param change: the amount to change the priority by
        :type change: float
        """
        if (not isinstance(idx, int) or idx < 0 or idx >= len(self._priority_tree)):
            raise ValueError("The index must be a valid index in the range of the tree.")
        if (not isinstance(change, float)):
            raise ValueError("The change must be a float.")
        
        # Add the change in priority to this node.
        self._priority_tree[idx] += change
        
        # If this is not the root node, propagate the change to the parent.
        if (idx > 0):
            self._propagate((idx - 1) // 2, change)
    
    def _sample_helper(self, idx: int, s: float):
        """
        Recursive helper function for sampling a transition based on s.
        Should only be called by PrioritizedReplayBuffer.sample.
        :param idx: is the current index being looked at.
        :type idx: int
        :param s: the prefix sum to search for.
        :type s: float
        :return: the index of a sampled transition.
        :rtype: int
        """ 
        idx_left = 2 * idx + 1

        if idx_left >= len(self._priority_tree):
            return self._to_transition_idx(idx)

        if s <= self._priority_tree[idx_left]:
            return self._sample_helper(idx_left, s)
        else:
            return self._sample_helper(idx_left + 1, s - self._priority_tree[idx_left])
    
    def _to_transition_idx(self, tree_idx: int):
        """
        Calculates the corresponding transition index of the given tree
        index.
        :param tree_idx: is the index for a node in this SumTree
        :type tree_idx: int
        :return: the index for the transition that corresponds to given
        tree node.
        :rtype: int
        """
        return tree_idx - self.max_length + 1
    
    def _to_tree_idx(self, idx: int):
        """
        Calculates the corresponding tree index of the given transition
        index.
        :param idx: is the index for a transition stored in this buffer.
        :type idx: int
        :return: the index for the tree node that corresponds to that
        transition.
        """
        return idx + self.max_length - 1
        
    def _to_transition_idx(self, tree_idx: int):
        """
        Calculates the corresponding transition index of the given tree
        index.
        :param tree_idx: is the index for a node in this SumTree
        :type tree_idx: int
        :return: the index for the transition that corresponds to given
        tree node.
        :rtype: int
        """
        return tree_idx - self.max_length + 1
   
class HindsightReplayBuffer(ReplayBuffer):
    """
    
    """
    def __init__(self, learner, max_length, empty_trans, history_length: int = 1):
        """
        
        """
        super().__init__(learner, max_length, empty_trans, history_length)
        self._hindsight_buffer = deque()
    
    def append_frame(self, transition_frame):
        """
        Appends a given framed to the buffer.
        :param transition_frame: the transition frame to append to the end
        of this buffer
        :type transition_frame: TransitionFrame
        """
        # Add the transition_frame to the transitions array.
        super().append_frame(transition_frame)
        self._hindsight_buffer.append(transition_frame)
        
    def apply_hindsight(self):
        """
        
        """
        goal = np.asarray(self._hindsight_buffer[-1].next_state)
        while self._hindsight_buffer:
            current = self._hindsight_buffer.popleft()
            state = np.asarray(current.state)
            reward = current.reward
            is_done = False
            if (np.sum(np.abs((state - goal))) == 0):
                reward = 0
                is_done = True
            
            if (isinstance(current, TransitionFrame)):
                super().append_frame(TransitionFrame(goal, -1, reward, goal, is_done))
            elif (isinstance(current, ActionTransitionFrame)):
                super().append_frame(ActionTransitionFrame(-1, goal, -1, reward, goal, is_done))
