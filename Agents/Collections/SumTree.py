import numpy as np

class SumTree:
    """
    SumTree, a variant of a Binary Tree, is a tree data structure where the
    value of a node is the sum of the values in the left and right
    subtrees.
    This SumTree is a variant specifically designed for Prioritized
    Experience Replay. The values are specifically the priority of the data
    and unlike normal SumTrees where all nodes can store data, only the
    leaf nodes store data.
    """
    def __init__(self, capacity: int):
        """
        Constructs a Priority SumTree.
        :param capacity: is the maximum capacity of the SumTree
        :type capacity: int
        """
        if (not isinstance(capacity, int) or capacity <= 0):
            raise ValueError("The capacity of the Tree must be a positive integer.")
        self._capacity = capacity
        self._cur_idx = 0
        self._transitions = np.zeros(self._capacity, dtype = object)
        self._size = 0
        '''
        Stores the priorities of the transitions. The leaf nodes, the
        second half tree array, are the direct priorities of the
        transitions and the parent nodes contain the sum of the priorities
        of all of their children.
        '''
        self._tree = np.zeros(2 * self._capacity - 1)
    
    def __len__(self):
        """
        Returns the size of this SumTree.
        :return: the size of the SumTree
        :rtype: int
        """
        return self._size
    
    def add(self, transition_frame, priority: float):
        """
        Adds the given transition frame to the SumTree with the given
        priority. If this SumTree is already at capacity, then the oldest
        entry is overwritten.
        :param transition_frame: is the the transition_frame to store in
        the SumTree
        :param priority: is the priority to give the transition_frame
        :type priority: float
        """
        # Add the transition_frame to the transitions array.
        self._transitions[self._cur_idx] = transition_frame
        
        # If the size is less than the capacity, then increment the size.
        if self._size < self._capacity:
            self._size += 1
        
        # Add/update the priority for this transition frame in the SumTree.
        self.update(self._cur_idx, priority)
        
        # Increment the current index.
        self._cur_idx = (self._cur_idx + 1) % self._capacity
    
    def sample(self, s: float):
        """
        Samples for a transition based on s.
        :param s: ?
        :type s: float
        :return: the index of a sampled transition.
        :rtype: int
        """
        if (not isinstance(s, float)):
            raise ValueError("The s must be a float.")
        
        return self._sample_helper(0, s)
    
    def sum_priority(self):
        """
        Returns the sum of priorities of all transitions in this SumTree.
        :return: the sum of all priorities
        :rtype: float
        """
        return self._tree[0]
    
    def update(self, idx: int, priority: float):
        """
        Updates the priority of the transition frame at the given idx.
        :param idx: is the index of the transition to update the priority of
        :type idx: int
        :param priority: is priority to give the transition
        :type priority: float
        """
        if (not isinstance(idx, int) or idx < 0 or idx >= self._size):
            raise ValueError("The index must be a valid index in the range of the current size of the SumTree.")
        if (not isinstance(priority, float)):
            raise ValueError("The priority must be a float.")
        
        # Update the priority of the given transition and propagate to
        # change to the parent nodes.
        tree_idx = self._to_tree_idx(idx)
        self._propagate(tree_idx, priority - self._tree[tree_idx])
    
    def _propagate(self, idx: int, change: float):
        """
        Recursively propogates the change in the priority of a node up the
        tree until the change is propagated to the root.
        :param idx: is the index for a node in the array to propagate the
        change to
        :type idx: int
        :param change: the amount to change the priority by
        :type change: float
        """
        if (not isinstance(idx, int) or idx < 0 or idx >= len(self._tree)):
            raise ValueError("The index must be a valid index in the range of the tree.")
        if (not isinstance(change, float)):
            raise ValueError("The change must be a float.")
        
        # Add the change in priority to this node.
        self._tree[idx] += change
        
        # If this is not the root node, propagate the change to the parent.
        if (idx > 0):
            self._propagate((idx - 1) // 2, change)
    
    def _sample_helper(self, idx: int, s: float):
        """
        Recursive helper function for sampling a transition based on s.
        Should only be called by SumTree.sample.
        :param idx: is the current index being looked at.
        :type idx: int
        :param s: ?
        :type s: float
        :return: the index of a sampled transition.
        :rtype: int
        """ 
        idx_left = 2 * idx + 1

        if idx_left >= len(self._tree):
            return idx

        if s <= self._tree[idx_left]:
            return self._sample_helper(idx_left, s)
        else:
            return self._sample_helper(idx_left + 1, s - self._tree[idx_left])
    
    def _to_tree_idx(self, trans_idx: int):
        """
        Calculates the corresponding tree index of the given transition
        index.
        :param trans_idx: is the index for transition stored in this
        SumTree
        :type trans_idx: int
        :return: the index for the tree node that corresponds to that
        transition.
        :rtype: int
        """
        return trans_idx + self._capacity - 1
    
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
        return tree_idx - self._capacity + 1
