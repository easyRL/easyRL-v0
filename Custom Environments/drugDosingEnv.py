import pickle
from random import random, randint

from interval import Interval

from Environments.environment import Environment

""" x1=N,x2=T,x3=I,x4=C
goal: x2=0 (&x1=1 in case3)"""
A1, A2, A3 = 0.2, 0.3, 0.1
B1, B2 = 1., 1.
C1, C2, C3, C4 = 1., 0.5, 1., 1.
D1, D2 = 0.2, 1.
R1, R2 = 1.5, 1.
S = 0.33
ALPHA = 0.3
RHO = 0.01

STATE12 = {Interval(-999, 0.0063): 1,
           Interval(0.0063, 0.0125): 2,
           Interval(0.0125, 0.025): 3,
           Interval(0.025, 0.01): 4,
           Interval(0.01, 0.05): 5,
           Interval(0.02, 0.1): 6,
           Interval(0.1, 0.2): 7,
           Interval(0.2, 0.25): 8,
           Interval(0.25, 0.3): 9,
           Interval(0.3, 0.35): 10,
           Interval(0.35, 0.4): 11,
           Interval(0.4, 0.45): 12,
           Interval(0.45, 0.5): 13,
           Interval(0.5, 0.55): 14,
           Interval(0.55, 0.6): 15,
           Interval(0.6, 0.65): 16,
           Interval(0.65, 0.7): 17,
           Interval(0.7, 0.8): 18,
           Interval(0.8, 0.9): 19,
           Interval(0.9, 999): 20}
STATE3 = {Interval(-999, 0.03): 1,
          Interval(0.03, 0.1): 2,
          Interval(0.1, 0.2): 3,
          Interval(0.2, 0.3): 4,
          Interval(0.3, 0.4): 5,
          Interval(0.4, 0.5): 6,
          Interval(0.5, 0.6): 7,
          Interval(0.6, 0.7): 8,
          Interval(0.7, 0.8): 9,
          Interval(0.8, 0.9): 10,
          Interval(0.9, 1.): 11,
          Interval(1., 1.2): 12,
          Interval(1.2, 1.4): 13,
          Interval(1.4, 1.6): 14,
          Interval(1.6, 1.8): 15,
          Interval(1.8, 2.): 16,
          Interval(2., 2.2): 17,
          Interval(2.2, 2.5): 18,
          Interval(2.5, 3.): 19,
          Interval(3., 999.): 20}
ACTION_N = (0., 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.)
ACTION_P = \
    (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.87, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.97, 0.98, 1.)
GOAL_N = 0.
GOAL_P = 0.15
U_MAX_N = 10
U_MAX_P = 0.5

MAX_ITER = 200


class DrugDosingEnv(Environment):
    displayName = 'DrugDosing'

    def __init__(self, case, first_stage=False):
        super().__init__()
        self.case = case
        self.first_stage = first_stage
        self.data = self.load_data()
        if case == 3:
            self.BETA = 0.9

    def step(self, action):
        s, x1, x2, x3, x4, t = self.state

        ek = self._get_error(x1, x2)
        x1, x2, x3, x4 = self._update_x(x1, x2, x3, x4, action, t)
        ek1 = self._get_error(x1, x2)
        s = self._get_state(ek1)

        new_state = (s, x1, x2, x3, x4, t + 1)
        reward = (ek - ek1) / ek if ek1 < ek else 0
        done = True if x1 <= 0 or x2 <= 0 or t >= MAX_ITER else False
        return new_state, reward, done

    def reset(self):
        self.done = False
        states = self.data[randint(0, len(self.data) - 1)]
        self.data = states[randint(0, len(states) - 1)]

    def load_data(self):
        if self.case == 1 or self.case == 3 or (self.case == 2 and self.first_stage is False):
            return pickle.load(open('./drugDosing_N', 'rb'))
        else:
            return pickle.load(open('./drugDosing_P', 'rb'))

    def _get_error(self, x1, x2):
        return x2 if self.case == 1 or self.case == 2 else self.BETA * x2 + (1 - self.BETA) * (1 - x1)

    def _get_state(self, ek):
        if self.case == 1 or self.case == 2:
            for i in STATE12:
                if ek in i:
                    return STATE12[i]
        else:
            for i in STATE3:
                if ek in i:
                    return STATE3[i]

    def _update_x(self, x1, x2, x3, x4, action, t):
        x1_u = R2 * x1 * (1 - B2 * x1) - C4 * x1 * x2 - A3 * x1 * x4
        x2_u = R1 * x2 * (1 - B1 * x2) - C2 * x3 * x2 - C3 * x2 * x1 - A2 * x2 * x4
        x3_u = S + (RHO * x3 * x2) / (ALPHA + x2) - C1 * x3 * x2 - D1 * x3 - A1 * x3 * x4
        if self.case == 1 or self.case == 3 or (self.case == 2 and t >= 91):
            x4_u = -D2 * x4 + U_MAX_N * ACTION_N[action]
        else:
            x4_u = -D2 * x4 + U_MAX_P * ACTION_P[action]
        return x1 + x1_u, x2 + x2_u, x3 + x3_u, x4 + x4_u


class CustomEnv(Environment):
    displayName = 'Custom_DrugDosing'

    def __init__(self, case):
        super().__init__()
        self.env = DrugDosingEnv(case)
        self.state = self.env.reset()
        self.done = False
        self.state_size = (6,)  # s,x1,x2,x3,x4,t
        self.action_size = 20

    def step(self, action):
        self.state, reward, self.done = self.env.step(action)
        return reward

    def reset(self):
        self.state = self.env.reset()
        self.done = False

    def sample_action(self):
        return randint(0, self.action_size - 1)

    def render(self):
        pass

    def close(self):
        pass


class DataGenerator(DrugDosingEnv):

    def generate(self, num, first_stage=False):
        scenarios = []
        for i in range(num):
            N, T, I, C, t = random(), random(), random(), 0, 1 if not (self.case == 2 and not first_stage) else 91
            s = self._get_state(self._get_error(N, T))
            self.state = s, N, T, I, C, t
            scenario = [self.state]

            for t in range(2 if not (self.case == 2 and not first_stage) else 92,
                           MAX_ITER + 1 if not (self.case == 2 and first_stage) else 91):
                action = randint(0, 19)
                self.state, _, done = self.step(action)
                scenario.append(self.state)
                if done:
                    break
            scenarios.append(scenario)
        return scenarios


def generate_all(num):
    data_N = DataGenerator(case=1).generate(num, first_stage=False)
    pickle.dump(data_N, open('./drugDosing_N', 'wb'))
    data_P = DataGenerator(case=2).generate(num, first_stage=True)
    pickle.dump(data_P, open('./drugDosing_P', 'wb'))

# generate_all(50000)
