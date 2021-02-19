from collections import namedtuple

TransitionFrame = namedtuple('TransitionFrame', ['state', 'action', 'reward', 'next_state', 'is_done'])
ActionTransitionFrame = namedtuple('ActionTransitionFrame', ['prev_action', 'state', 'action', 'reward', 'next_state', 'is_done'])
