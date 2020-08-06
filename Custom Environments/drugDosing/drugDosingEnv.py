import pickle
from copy import deepcopy
from random import randint, uniform, random, seed

from interval import Interval

from Environments.environment import Environment

STATE12 = {Interval(-999, 0.0063): 1,
           Interval(0.0063, 0.0125): 2,
           Interval(0.0125, 0.025): 3,
           Interval(0.025, 0.01): 4,
           Interval(0.01, 0.05): 5,
           Interval(0.05, 0.1): 6,
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
U_MAX_N = 10.
U_MAX_P = 0.5

# x1=N,x2=T,x3=I,x4=C
# goal: x2=0 (& x1=1 in case3)
A1, A2, A3 = 0.2, 0.3, 0.1
B1, B2 = 1., 1.
C1, C2, C3, C4 = 1., 0.5, 1., 1.
D1, D2 = 0.2, 1.
R1, R2 = 1.5, 1.
S = 0.33
ALPHA = 0.3
RHO = 0.01
BETA = 0.8

# Default setting (1: young patient, 2: young pregnant woman, 3: elderly patient)
CASE = 1

MAX_ITER = 200


# Only case=1 is mentioned in the paper
# Also lower bound is added to avoid bizarre result (not mentioned in paper)
class Patient:
    def __init__(self, default=True, case=1, is_first_stage=False, x1=None, x2=None, x3=None, x4=None, t=1, s=None):
        if s:
            seed(s)
        self.case = case
        self.is_first_stage = is_first_stage
        self.t = t

        self.A2 = A2 if default else uniform(.25, .5)
        self.A1 = A1 if default else uniform(0.1, self.A2)
        self.A3 = A3 if default else uniform(0.1, self.A1)
        self.B2 = B2 if default else 1.
        self.B1 = B1 if default else uniform(1., 1.5)
        self.C1 = C1 if default else uniform(.3, 1.)
        self.C2 = C2 if default else uniform(.3, 1.)
        self.C3 = C3 if default else uniform(.3, 1.)
        self.C4 = C4 if default else uniform(.3, 1.)
        self.D1 = D1 if default else uniform(.15, .3)
        self.D2 = D2 if default else 1.
        self.R1 = R1 if default else uniform(1.2, 1.6)
        self.R2 = R2 if default else 1.
        self.S = S if default else uniform(.3, .5)
        self.ALPHA = ALPHA if default else uniform(.3, .5)
        self.RHO = RHO if default else uniform(.01, .05)

        self.x1 = x1 if x1 is not None else R2 if default else 0.6
        self.x2 = x2 if x2 is not None else 1. - self.x1
        self.x3 = x3 if x3 is not None else uniform(.1, .2)
        self.x4 = x4 if x4 is not None else 0.
        self.x20 = self.x2
        self.s = self._get_state(self._get_error())

    def update(self, action):
        ek = self._get_error()
        x1_u = self.R2 * self.x1 * (1 - self.B2 * self.x1) - self.C4 * self.x1 * self.x2 - self.A3 * self.x1 * self.x4
        x2_u = self.R1 * self.x2 * (
                1 - self.B1 * self.x2) - self.C2 * self.x3 * self.x2 - self.C3 * self.x2 * self.x1 - self.A2 * self.x2 * self.x4
        x3_u = self.S + (self.RHO * self.x3 * self.x2) / (
                self.ALPHA + self.x2) - self.C1 * self.x3 * self.x2 - self.D1 * self.x3 - self.A1 * self.x3 * self.x4
        if self.case in (1, 3) or (self.case == 2 and not self.is_first_stage):
            x4_u = -self.D2 * self.x4 + U_MAX_N * ACTION_N[action]
        else:
            x4_u = -self.D2 * self.x4 + U_MAX_P * ACTION_P[action]
        self.x1 = max(self.x1 + x1_u, 0.)
        self.x2 = max(self.x2 + x2_u, 0.)
        self.x3 = max(self.x3 + x3_u, 0.)
        self.x4 = max(self.x4 + x4_u, 0.)
        ek_new = self._get_error()

        self.s = self._get_state(ek_new)
        self.t = self.t + 1
        reward = self._get_reward(ek, ek_new)
        done = self.x2 <= 1e-3 or self.x1 <= 1e-3 or self.t > MAX_ITER
        cured = True if self.x2 <= 1e-3 and self.x1 >= 0.999 else False
        dead = True if self.x1 <= 1e-3 else False
        return self.s, reward, done, cured, dead, self._get_N(), self._get_T()

    def _get_error(self):
        return max(0., self.x2 - 0. if self.case in (1, 2) else BETA * self.x2 + (1. - BETA) * (1. - self.x1))

    def _get_state(self, ek):
        if self.case in (1, 2):
            for i in STATE12:
                if ek in i:
                    return STATE12[i]
        else:
            for i in STATE3:
                if ek in i:
                    return STATE3[i]

    @staticmethod
    def _get_reward(ek, ek_new):
        return float(ek - ek_new) / (ek + 1e-5) if ek_new < ek else 0.

    def _get_N(self):
        return abs(self.x1 - 1) * 100

    def _get_T(self):
        return self.x2 / (self.x20 + 1e-5) * 100


class DrugDosingEnv(Environment):
    displayName = 'DrugDosing'

    def __init__(self, case, is_first_stage):
        super().__init__()
        self.case = case
        self.is_first_stage = is_first_stage
        self.data = self.load_data()
        self.patient = None
        # For testing
        self.agent = None
        self.patients_test = None

    def step(self, action):
        result = self.patient.update(action)
        return result[0], result[1], result[2]

    def reset(self):
        datas = self.data[randint(0, len(self.data) - 1)]
        data = datas[randint(0, len(datas) - 2)]
        self.patient = Patient(case=self.case, x1=data[0], x2=data[1], x3=data[2], x4=data[3], t=data[4])
        return self.patient.s

    def load_data(self):
        if self.case in (1, 3) or (self.case == 2 and not self.is_first_stage):
            return pickle.load(open('./Custom Environments/drugDosing/drugDosing_N', 'rb'))
        else:
            return pickle.load(open('./Custom Environments/drugDosing/drugDosing_P', 'rb'))

    def set_agent(self, agent):
        self.agent = agent
        if self.case == 1:
            self.patients_test = [Patient(default=False, case=1) for _ in range(15)]

    # Advanced feature, only avilable when importing agent
    def episode_finish(self, episode):
        if self.agent:
            if episode and not episode % 500:
                self.agent.alpha *= 0.8
            if episode and not episode % 1000 and self.case == 1:
                self.print_NT()

    def print_NT(self):
        Ns, N1, N4, N7 = [], [], [], []
        Ts, T1, T4, T7 = [], [], [], []
        for p in self.patients_test:
            t = 0
            p = deepcopy(p)
            T_added, N_added, p_cured, p_dead = False, False, False, False
            qs, ss, x1s, x2s, = [], [], [], []
            while t < MAX_ITER:
                t += 1
                q = self.agent.choose_action(p.s)
                ss.append(p.s)
                x1s.append(p.x1)
                x2s.append(p.x2)
                qs.append(q)
                s, reward, done, cured, dead, N, T = p.update(q)
                if t == 7:
                    N1.append(N)
                    T1.append(T)
                elif t == 28:
                    N4.append(N)
                    T4.append(T)
                elif t == 49:
                    N7.append(N)
                    T7.append(T)
                if N <= 1e-5 and not N_added:
                    Ns.append(t)
                    N_added = True
                if T <= 1e-5 and not T_added:
                    Ts.append(t)
                    T_added = True
                if dead:
                    p_dead = True
                    print('dead!')
                    # return
                    break
                if cured:
                    if t < 7:
                        N1.append(0)
                        T1.append(0)
                    if t < 28:
                        N4.append(0)
                        N4.append(0)
                    if t < 49:
                        N7.append(0)
                        T7.append(0)
                    p_cured = True
                    print('cured!')
                    break
            if not p_cured and not p_dead:
                pass
                print('survied!')
                # return
            # print(qs)
            # print(ss)
            # print(x1s)
            # print(x2s)
        if len(Ns) + len(Ts) != 30:
            print('')
            return
        print('Number of days to achieve the target value: N_avg:' + str(sum(Ns) / len(Ns)))
        print('Number of days to achieve the target value: N_max:' + str(max(Ns)))
        print('Number of days to achieve the target value: N_min:' + str(min(Ns)))
        print('Number of days to achieve the target value: T_avg:' + str(sum(Ts) / len(Ts)))
        print('Number of days to achieve the target value: T_max:' + str(max(Ts)))
        print('Number of days to achieve the target value: T_min:' + str(min(Ts)))

        print('Percent value; after 1 week of chemotherapy: N_avg:' + str(sum(N1) / len(N1)))
        print('Percent value; after 1 week of chemotherapy: N_max:' + str(max(N1)))
        print('Percent value; after 1 week of chemotherapy: N_min:' + str(min(N1)))
        print('Percent value; after 1 week of chemotherapy: T_avg:' + str(sum(T1) / len(T1)))
        print('Percent value; after 1 week of chemotherapy: T_max:' + str(max(T1)))
        print('Percent value; after 1 week of chemotherapy: T_min:' + str(min(T1)))

        print('Percent value; after 4 week of chemotherapy: N_avg:' + str(sum(N4) / len(N4)))
        print('Percent value; after 4 week of chemotherapy: N_max:' + str(max(N4)))
        print('Percent value; after 4 week of chemotherapy: N_min:' + str(min(N4)))
        print('Percent value; after 4 week of chemotherapy: T_avg:' + str(sum(T4) / len(T4)))
        print('Percent value; after 4 week of chemotherapy: T_max:' + str(max(T4)))
        print('Percent value; after 4 week of chemotherapy: T_min:' + str(min(T4)))

        print('Percent value; after 7 week of chemotherapy: N_avg:' + str(sum(N7) / len(N7)))
        print('Percent value; after 7 week of chemotherapy: N_max:' + str(max(N7)))
        print('Percent value; after 7 week of chemotherapy: N_min:' + str(min(N7)))
        print('Percent value; after 7 week of chemotherapy: T_avg:' + str(sum(T7) / len(T7)))
        print('Percent value; after 7 week of chemotherapy: T_max:' + str(max(T7)))
        print('Percent value; after 7 week of chemotherapy: T_min:' + str(min(T7)))


class CustomEnv(Environment):
    displayName = 'Custom_DrugDosing'

    def __init__(self, case=CASE, is_first_stage=False):
        super().__init__()
        self.env = DrugDosingEnv(case, is_first_stage)
        self.state = self.env.reset()
        self.done = False
        self.state_size = (20,)
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


def generate(num, case, is_first_stage):
    scenarios = []
    for i in range(num):
        # assumption
        N = random()
        T, I, C, t = 1 - N, uniform(.1, .2), 0., 1 if not (case == 2 and not is_first_stage) else 91
        p = Patient(case=case, is_first_stage=is_first_stage, x1=N, x2=T, x3=I, x4=C, t=t)
        scenario = [[p.x1, p.x2, p.x3, p.x4, p.t]]
        for t in range(2 if not (case == 2 and not is_first_stage) else 92,
                       MAX_ITER + 1 if not (case == 2 and is_first_stage) else 91):
            action = randint(0, 19)
            _, _, done, _, _, _, _ = p.update(action)
            scenario.append([p.x1, p.x2, p.x3, p.x4, p.t])
            if done:
                break
        scenarios.append(scenario)
    return scenarios


def generate_all(num):
    data_N = generate(num, case=1, is_first_stage=False)
    pickle.dump(data_N, open('./drugDosing_N', 'wb'))
    print('Normal data generated')
    data_P = generate(num, case=2, is_first_stage=True)
    pickle.dump(data_P, open('./drugDosing_P', 'wb'))
    print('Pregnant data generated')

# Uncomment next line to generate data for initialization
# generate_all(50000)
