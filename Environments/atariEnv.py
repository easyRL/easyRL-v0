from Environments import environment
import gym
from PIL import Image
import cv2
import numpy as np
from abc import ABC
import random

class AtariEnv(environment.Environment, ABC):
    displayName = 'AtariEnv'
    subEnvs = []

    def __init__(self):
        self.image_width = 84
        self.image_height = 84
        self.state_size = (self.image_width, self.image_height, 1)
        self.env = None
        self.state = None
        self.rawImg = None
        self.done = None
        self.total_rewards = None

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        self.state = self.preprocess(observation)
        return reward

    def reset(self):
        self.rawImg = self.env.reset()
        self.state = self.preprocess(self.rawImg)
        self.done = False
        self.total_rewards = 0

    def sample_action(self):
        return self.env.action_space.sample()

    def preprocess(self, image):
        self.rawImg = image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return np.reshape(image, self.state_size)

    def render(self, mode='RGB'):
        return Image.fromarray(self.rawImg.astype('uint8'), 'RGB')


class adventureEnv(AtariEnv):
    displayName = 'adventure'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Adventure-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(adventureEnv)


class air_raidEnv(AtariEnv):
    displayName = 'air_raid'

    def __init__(self):
        super().__init__()
        self.env = gym.make('AirRaid-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(air_raidEnv)


class alienEnv(AtariEnv):
    displayName = 'alien'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Alien-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(alienEnv)


class amidarEnv(AtariEnv):
    displayName = 'amidar'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Amidar-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(amidarEnv)


class assaultEnv(AtariEnv):
    displayName = 'assault'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Assault-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(assaultEnv)


class asterixEnv(AtariEnv):
    displayName = 'asterix'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Asterix-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(asterixEnv)


class asteroidsEnv(AtariEnv):
    displayName = 'asteroids'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Asteroids-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(asteroidsEnv)


class atlantisEnv(AtariEnv):
    displayName = 'atlantis'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Atlantis-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(atlantisEnv)


class bank_heistEnv(AtariEnv):
    displayName = 'bank_heist'

    def __init__(self):
        super().__init__()
        self.env = gym.make('BankHeist-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(bank_heistEnv)


class battle_zoneEnv(AtariEnv):
    displayName = 'battle_zone'

    def __init__(self):
        super().__init__()
        self.env = gym.make('BattleZone-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(battle_zoneEnv)


class beam_riderEnv(AtariEnv):
    displayName = 'beam_rider'

    def __init__(self):
        super().__init__()
        self.env = gym.make('BeamRider-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(beam_riderEnv)


class berzerkEnv(AtariEnv):
    displayName = 'berzerk'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Berzerk-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(berzerkEnv)


class bowlingEnv(AtariEnv):
    displayName = 'bowling'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Bowling-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(bowlingEnv)


class boxingEnv(AtariEnv):
    displayName = 'boxing'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Boxing-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(boxingEnv)


class breakoutEnv(AtariEnv):
    displayName = 'breakout'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Breakout-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(breakoutEnv)


class carnivalEnv(AtariEnv):
    displayName = 'carnival'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Carnival-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(carnivalEnv)


class centipedeEnv(AtariEnv):
    displayName = 'centipede'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Centipede-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(centipedeEnv)


class chopper_commandEnv(AtariEnv):
    displayName = 'chopper_command'

    def __init__(self):
        super().__init__()
        self.env = gym.make('ChopperCommand-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(chopper_commandEnv)


class crazy_climberEnv(AtariEnv):
    displayName = 'crazy_climber'

    def __init__(self):
        super().__init__()
        self.env = gym.make('CrazyClimber-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(crazy_climberEnv)


class demon_attackEnv(AtariEnv):
    displayName = 'demon_attack'

    def __init__(self):
        super().__init__()
        self.env = gym.make('DemonAttack-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(demon_attackEnv)


class double_dunkEnv(AtariEnv):
    displayName = 'double_dunk'

    def __init__(self):
        super().__init__()
        self.env = gym.make('DoubleDunk-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(double_dunkEnv)


class elevator_actionEnv(AtariEnv):
    displayName = 'elevator_action'

    def __init__(self):
        super().__init__()
        self.env = gym.make('ElevatorAction-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(elevator_actionEnv)


class enduroEnv(AtariEnv):
    displayName = 'enduro'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Enduro-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(enduroEnv)


class fishing_derbyEnv(AtariEnv):
    displayName = 'fishing_derby'

    def __init__(self):
        super().__init__()
        self.env = gym.make('FishingDerby-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(fishing_derbyEnv)


class freewayEnv(AtariEnv):
    displayName = 'freeway'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Freeway-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(freewayEnv)


class frostbiteEnv(AtariEnv):
    displayName = 'frostbite'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Frostbite-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(frostbiteEnv)


class gopherEnv(AtariEnv):
    displayName = 'gopher'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Gopher-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(gopherEnv)


class gravitarEnv(AtariEnv):
    displayName = 'gravitar'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Gravitar-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(gravitarEnv)


class heroEnv(AtariEnv):
    displayName = 'hero'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Hero-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(heroEnv)


class ice_hockeyEnv(AtariEnv):
    displayName = 'ice_hockey'

    def __init__(self):
        super().__init__()
        self.env = gym.make('IceHockey-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(ice_hockeyEnv)


class jamesbondEnv(AtariEnv):
    displayName = 'jamesbond'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Jamesbond-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(jamesbondEnv)


class journey_escapeEnv(AtariEnv):
    displayName = 'journey_escape'

    def __init__(self):
        super().__init__()
        self.env = gym.make('JourneyEscape-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(journey_escapeEnv)


class kangarooEnv(AtariEnv):
    displayName = 'kangaroo'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Kangaroo-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(kangarooEnv)


class krullEnv(AtariEnv):
    displayName = 'krull'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Krull-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(krullEnv)


class kung_fu_masterEnv(AtariEnv):
    displayName = 'kung_fu_master'

    def __init__(self):
        super().__init__()
        self.env = gym.make('KungFuMaster-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(kung_fu_masterEnv)


class montezuma_revengeEnv(AtariEnv):
    displayName = 'montezuma_revenge'

    def __init__(self):
        super().__init__()
        self.env = gym.make('MontezumaRevenge-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(montezuma_revengeEnv)


class ms_pacmanEnv(AtariEnv):
    displayName = 'ms_pacman'

    def __init__(self):
        super().__init__()
        self.env = gym.make('MsPacman-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(ms_pacmanEnv)


class name_this_gameEnv(AtariEnv):
    displayName = 'name_this_game'

    def __init__(self):
        super().__init__()
        self.env = gym.make('NameThisGame-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(name_this_gameEnv)


class phoenixEnv(AtariEnv):
    displayName = 'phoenix'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Phoenix-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(phoenixEnv)


class pitfallEnv(AtariEnv):
    displayName = 'pitfall'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Pitfall-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(pitfallEnv)


class pongEnv(AtariEnv):
    displayName = 'pong'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Pong-v0')
        self.action_size = 2

    def step(self, action):
        if action == 0:
            action = 2
        else:
            action = 5
        return super().step(action)

    def sample_action(self):
        return random.randrange(self.action_size)

AtariEnv.subEnvs.append(pongEnv)


class pooyanEnv(AtariEnv):
    displayName = 'pooyan'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Pooyan-v0')
        self.action_size = self.env.action_space.n

0
AtariEnv.subEnvs.append(pooyanEnv)


class private_eyeEnv(AtariEnv):
    displayName = 'private_eye'

    def __init__(self):
        super().__init__()
        self.env = gym.make('PrivateEye-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(private_eyeEnv)


class qbertEnv(AtariEnv):
    displayName = 'qbert'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Qbert-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(qbertEnv)


class riverraidEnv(AtariEnv):
    displayName = 'riverraid'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Riverraid-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(riverraidEnv)


class road_runnerEnv(AtariEnv):
    displayName = 'road_runner'

    def __init__(self):
        super().__init__()
        self.env = gym.make('RoadRunner-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(road_runnerEnv)


class robotankEnv(AtariEnv):
    displayName = 'robotank'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Robotank-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(robotankEnv)


class seaquestEnv(AtariEnv):
    displayName = 'seaquest'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Seaquest-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(seaquestEnv)


class skiingEnv(AtariEnv):
    displayName = 'skiing'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Skiing-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(skiingEnv)


class solarisEnv(AtariEnv):
    displayName = 'solaris'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Solaris-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(solarisEnv)


class space_invadersEnv(AtariEnv):
    displayName = 'space_invaders'

    def __init__(self):
        super().__init__()
        self.env = gym.make('SpaceInvaders-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(space_invadersEnv)


class star_gunnerEnv(AtariEnv):
    displayName = 'star_gunner'

    def __init__(self):
        super().__init__()
        self.env = gym.make('StarGunner-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(star_gunnerEnv)


class tennisEnv(AtariEnv):
    displayName = 'tennis'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Tennis-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(tennisEnv)


class time_pilotEnv(AtariEnv):
    displayName = 'time_pilot'

    def __init__(self):
        super().__init__()
        self.env = gym.make('TimePilot-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(time_pilotEnv)


class tutankhamEnv(AtariEnv):
    displayName = 'tutankham'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Tutankham-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(tutankhamEnv)


class up_n_downEnv(AtariEnv):
    displayName = 'up_n_down'

    def __init__(self):
        super().__init__()
        self.env = gym.make('UpNDown-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(up_n_downEnv)


class ventureEnv(AtariEnv):
    displayName = 'venture'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Venture-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(ventureEnv)


class video_pinballEnv(AtariEnv):
    displayName = 'video_pinball'

    def __init__(self):
        super().__init__()
        self.env = gym.make('VideoPinball-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(video_pinballEnv)


class wizard_of_worEnv(AtariEnv):
    displayName = 'wizard_of_wor'

    def __init__(self):
        super().__init__()
        self.env = gym.make('WizardOfWor-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(wizard_of_worEnv)


class yars_revengeEnv(AtariEnv):
    displayName = 'yars_revenge'

    def __init__(self):
        super().__init__()
        self.env = gym.make('YarsRevenge-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(yars_revengeEnv)


class zaxxonEnv(AtariEnv):
    displayName = 'zaxxon'

    def __init__(self):
        super().__init__()
        self.env = gym.make('Zaxxon-v0')
        self.action_size = self.env.action_space.n


AtariEnv.subEnvs.append(zaxxonEnv)
