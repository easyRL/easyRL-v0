from abc import ABC
import random
from enum import Enum
from typing import List
from PIL import Image, ImageDraw

import gym

from Environments import environment

GOOD_SELLER_GOOD_SELL_CHANCE = 1
BAD_SELLER_GOOD_SELL_CHANCE = 0
GOOD_ADVISER_HONEST_CHANCE = 1
BAD_ADVISER_HONEST_CHANCE = 0
SATISFACTORY_OUTCOME_REWARD = 100
UNSATISFACTORY_OUTCOME_REWARD = -100
ADVISER_QUERY_REWARD = -1
SELLER_QUERY_REWARD = -10

class SaleEnv(gym.Env):
    displayName = 'SALE-POMDP'
    metadata = {'render.modes': []}

    def __init__(self, tie_settler_func=None, actors: int = 5, seller_prop: float = 0.2, good_sell_prop: float = 0.5, good_adviser_prop: float = 0.8, *penalty_functions):
        if tie_settler_func is None:
            tie_settler_func = round_with_minimum_one_tend_first
        self.ACTORS = actors
        self.SELLERS, self.ADVISERS = tie_settler_func(actors * seller_prop, actors * (1 - seller_prop))
        self.GOOD_SELLERS, self.BAD_SELLERS = tie_settler_func(self.SELLERS * good_sell_prop, self.SELLERS * (1 - good_sell_prop))
        self.GOOD_ADVISERS, self.BAD_ADVISERS = tie_settler_func(self.ADVISERS * good_adviser_prop, self.ADVISERS * (1 - good_adviser_prop))
        self.NUM_QUERY_SELLER_ACTIONS = self.SELLERS * self.ADVISERS
        self.NUM_QUERY_ADVISER_ACTIONS = self.ADVISERS ** 2
        self.NUM_BUY_ACTIONS = self.SELLERS
        self.NUM_DNB_ACTIONS = 1
        self.NUM_ACTIONS = self.NUM_QUERY_SELLER_ACTIONS + self.NUM_QUERY_ADVISER_ACTIONS + self.NUM_BUY_ACTIONS + self.NUM_DNB_ACTIONS

        self.observation_space = gym.spaces.Discrete(6)
        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)

        self.penalty_functions = penalty_functions

    def step(self, action: int) -> (object, float, bool, dict):
        total_rewards = 0
        for function in self.penalty_functions:
            total_rewards += function(action)

        if action < self.NUM_QUERY_SELLER_ACTIONS:  # Seller Query
            consulted_adviser_index = action % self.ADVISERS
            consulted_adviser = self.advisers[consulted_adviser_index]
            seller_in_question_index = int(action / self.ADVISERS)
            seller_in_question = self.sellers[seller_in_question_index]
            total_rewards += SELLER_QUERY_REWARD
            return consulted_adviser.advise_on_seller(seller_in_question).value, total_rewards, False, {'actor_number': seller_in_question_index, 'adviser_index': self.SELLERS + consulted_adviser_index, 'state': self.__get_state()}
        elif action < self.NUM_QUERY_SELLER_ACTIONS + self.NUM_QUERY_ADVISER_ACTIONS:  # Adviser Query
            consulted_adviser_index = (action - self.NUM_QUERY_SELLER_ACTIONS) % self.ADVISERS
            consulted_adviser = self.advisers[consulted_adviser_index]
            adviser_in_question_index = int((action - self.NUM_QUERY_SELLER_ACTIONS) / self.ADVISERS)
            adviser_in_question = self.advisers[adviser_in_question_index]
            total_rewards += ADVISER_QUERY_REWARD
            return consulted_adviser.advise_on_adviser(adviser_in_question).value, total_rewards, False, {'actor_number': self.SELLERS + adviser_in_question_index, 'adviser_index': self.SELLERS + consulted_adviser_index, 'state': self.__get_state()}
        elif action < self.NUM_QUERY_SELLER_ACTIONS + self.NUM_QUERY_ADVISER_ACTIONS + self.NUM_BUY_ACTIONS:  # Buy from seller
            chosen_seller = self.sellers[action - (self.NUM_QUERY_SELLER_ACTIONS + self.NUM_QUERY_ADVISER_ACTIONS)]
            outcome = chosen_seller.sell_product()
            if outcome:
                total_rewards += SATISFACTORY_OUTCOME_REWARD
            else:
                total_rewards += UNSATISFACTORY_OUTCOME_REWARD
            self.reset()
            return Observation.ended.value, total_rewards, True, {'state': self.__get_state()}
        elif action < self.NUM_ACTIONS:  # DNB
            def __check_sellers(index: int, seller_list: List[Seller]) -> bool:
                if index < len(seller_list):
                    return seller_list[index].good and __check_sellers(index + 1, seller_list)
            good_sellers_exist = __check_sellers(0, self.sellers)
            if good_sellers_exist:
                total_rewards += SATISFACTORY_OUTCOME_REWARD
            else:
                total_rewards += UNSATISFACTORY_OUTCOME_REWARD
            self.reset()
            return Observation.ended.value, total_rewards, True, {'state': self.__get_state()}
        else:  # Bad error argument
            # TODO: Error
            pass

    def reset(self):
        self.__generate_random_state()
        return Observation.none.value

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def __generate_random_state(self):
        self.sellers = []
        self.advisers = []
        good_seller = Seller(True, GOOD_SELLER_GOOD_SELL_CHANCE)
        bad_seller = Seller(False, BAD_SELLER_GOOD_SELL_CHANCE)
        good_adviser = Adviser(True, GOOD_ADVISER_HONEST_CHANCE)
        bad_adviser = Adviser(False, BAD_ADVISER_HONEST_CHANCE)
        for _ in range(self.GOOD_SELLERS):
            self.sellers.append(good_seller)
        for _ in range(self.BAD_SELLERS):
            self.sellers.append(bad_seller)
        for _ in range(self.GOOD_ADVISERS):
            self.advisers.append(good_adviser)
        for _ in range(self.BAD_ADVISERS):
            self.advisers.append(bad_adviser)
        random.shuffle(self.sellers)
        random.shuffle(self.advisers)

    def __get_state(self) -> list:
        state = []
        for seller in self.sellers:
            state.append(seller.good)
        for adviser in self.advisers:
            state.append(adviser.trustworthy)
        return state


# Observation enums to be returned by observation function
# Use Observation.<enum>.value to get associated number to convert to vector
class Observation(Enum):
    good = 0
    bad = 1
    trustworthy = 2
    untrustworthy = 3
    ended = 4
    none = 5


class Seller:
    def __init__(self, good: bool, good_prop: float):
        self.good = good
        self.good_prop = good_prop

    # Where True is a good product
    def sell_product(self) -> bool:
        return random.random() < self.good_prop

    def __repr__(self):
        return "Seller[Good Prop: " + str(self.good_prop) + "]"


class Adviser:
    def __init__(self, trustworthy: bool, right_prop: float):
        self.trustworthy = trustworthy
        self.right_prop = right_prop

    # Where True is a good adviser
    def advise_on_adviser(self, other_adviser: 'Adviser') -> Observation:
        tell_truth = random.random() < self.right_prop
        other_trust = other_adviser.trustworthy
        if (not tell_truth or other_trust) and (tell_truth or not other_trust):
            return Observation.trustworthy
        else:
            return Observation.untrustworthy

    # Where True is a good seller
    def advise_on_seller(self, other_seller: Seller) -> Observation:
        tell_truth = random.random() < self.right_prop
        other_good = other_seller.good
        if (not tell_truth or other_good) and (tell_truth or not other_good):
            return Observation.good
        else:
            return Observation.bad

    def __repr__(self):
        return "Adviser[Right Prop: " + str(self.right_prop) + ", Trustworthy: " + str(self.trustworthy) + "]"


# Will round each to the nearest int, giving ties to the former variable.
# The first will also be rounded up to 2 (granted that b can be rounded down the same)
# The first will always be rounded up to 1 and b to 0 if fa + fb = 1
def round_with_minimum_one_tend_first(float_a, float_b):
    a = round(float_a)
    b = round(float_b)
    if b == 0:  # Make b at least 1
        b += 1
        if a + b > float_a + float_b:
            a -= 1
    if a == 0:  # Make a at least 1. Overrides b's at least 1 above
        a += 1
        if a + b > float_a + float_b:
            b -= 1
    if a == 1 and b > 1:  # Make a at least 2 so long as b is at least 1
        a += 1
        if a + b > float_a + float_b:
            b -= 1
    if a + b < float_a + float_b:  # Increase a by 1 if by rounding chance the sum is too low
        a += 1
    if a + b > float_a + float_b:  # Decrease b by 1 if by rounding chance the sum is too high
        b -= 1
    return a, b


class CustomEnv(environment.Environment):
    displayName = 'Custom-SALE-POMDP'

    def __init__(self):
        super().__init__()
        self.env = SaleEnv()
        self.state = self.env.reset()
        self.done = False
        self.total_rewards = 0
        self.state_size = (6,)
        self.action_size = 18

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        self.state = to_one_hot(observation, len(Observation))
        return reward

    def reset(self):
        self.state = to_one_hot(self.env.reset(), len(Observation))
        self.done = False
        self.total_rewards = 0

    def render(self):
        return Image.new('RGB', (1, 1), 'white')

    def sample_action(self):
        return self.env.action_space.sample()

def to_one_hot(index: int, size: int):
    return [1 if x is index else 0 for x in range(size)]
