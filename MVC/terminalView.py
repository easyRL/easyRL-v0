from Agents import qLearning, drqn, deepQ, adrqn, agent, doubleDuelingQNative, drqnNative, drqnConvNative, ppoNative, reinforceNative, actorCriticNative, cem, npg, ddpg, sac, trpo, rainbow
from Environments import cartPoleEnv, cartPoleEnvDiscrete, atariEnv, frozenLakeEnv, pendulumEnv, acrobotEnv, mountainCarEnv
from MVC.model import Model
from Agents.sarsa import sarsa
import time, os

class View:
    agents = [qLearning.QLearning, sarsa, deepQ.DeepQ, deepQ.DeepQPrioritized, deepQ.DeepQHindsight, drqn.DRQN, drqn.DRQNPrioritized, drqn.DRQNHindsight, adrqn.ADRQN, adrqn.ADRQNPrioritized, adrqn.ADRQNHindsight, npg.NPG, ddpg.DDPG, cem.CEM, sac.SAC, trpo.TRPO, rainbow.Rainbow]
    environments = [cartPoleEnv.CartPoleEnv, cartPoleEnvDiscrete.CartPoleEnvDiscrete, frozenLakeEnv.FrozenLakeEnv,
                    pendulumEnv.PendulumEnv, acrobotEnv.AcrobotEnv, mountainCarEnv.MountainCarEnv]
    environments += atariEnv.AtariEnv.subEnvs

    def __init__(self, listener):
        self.listener = listener
        self.isHalted = False
        self.isTrained = False
        self.episodeNum = None
        self.dataPoints = []
        self.episodeStates = []
        self.paramValues = None

        self.environment = self.chooseEnvironment()
        self.agentType = self.chooseAgent()

        while not self.isHalted:
            self.mainMenu()

    def chooseEnvironment(self):
        count = 1
        valid = False
        while not valid:
            text = '\nChoose an environment:\n'
            for env in View.environments:
                text += str(count) + ') ' + env.displayName + '\n'
                count += 1
            choice = input(text)
            try:
                choice = int(choice)
                if 1 <= choice <= len(View.environments):
                    self.listener.setEnvironment(0, View.environments[choice-1])
                    valid = True
            except ValueError:
                print('Input must be an integer')
                pass

    def chooseAgent(self):
        count = 1
        valid = False
        selectedAgent = None
        while not valid:
            text = '\nChoose an agent:\n'
            for agent in View.agents:
                text += str(count) + ') ' + agent.displayName + '\n'
                count += 1
            choice = input(text)
            try:
                choice = int(choice)
                if 1 <= choice <= len(View.agents):
                    selectedAgent = View.agents[choice - 1]
                    self.listener.setAgent(0, selectedAgent)
                    valid = True
            except ValueError:
                print('Input must be an integer')
                pass
        return selectedAgent

    def chooseParameters(self):
        params = []
        paramValues = []
        params.append(agent.Agent.Parameter('Total Episodes', 1, float('inf'), 1, 1000, True, True, 'The number of episodes to train the agent'))
        params.append(agent.Agent.Parameter('Max Steps', 1, float('inf'), 1, 200, True, True, 'The maximum number of steps per episodes'))
        params += self.agentType.parameters
        count = 1
        print('Choose agent hyperparameters:')
        for param in params:
            text = '\n' + str(count) + ') ' + param.name + ' (' + param.toolTipText + ')\n'
            text += 'must be between ' + str(param.min) + ' and ' + str(param.max) + '\n'
            count += 1
            valid = False
            while not valid:
                choice = input(text)
                try:
                    choice = int(choice) if param.resolution % 1 == 0 else float(choice)
                    if param.min <= choice <= param.max:
                        paramValues.append(choice)
                        valid = True
                    else:
                        print('Input not within range')
                except ValueError:
                    print('Input must be a number')
                    pass
        return paramValues

    def mainMenu(self):
        count = 1
        text = '\n' + str(count) + ') Start training\n'
        count += 1
        text += str(count) + ') Load saved agent\n'
        count += 1
        if self.isTrained:
            text += str(count) + ') Start testing\n'
            count += 1
            text += str(count) + ') Save trained agent\n'
            count += 1
        text += str(count) + ') Exit\n'
        count += 1

        valid = False
        choice = None
        while not valid:
            choice = input(text)
            try:
                choice = int(choice)
                if 1 <= choice <= count-1:
                    valid = True
                else:
                    print('Input not within range')
            except ValueError:
                print('Input must be an integer')
                pass

        if choice == count-1:
            self.isHalted = True
        elif choice == 1:
            self.paramValues = self.chooseParameters()
            self.listener.startTraining(0, self.paramValues)
            self.isTrained = True
            self.episodeNum = 0
            self.dataPoints.clear()
            self.episodeStates.clear()
            while self.checkMessages():
                time.sleep(0.1)
        elif choice == 2:
            workingDir = os.getcwd()
            loaded = False
            while not loaded:
                text = 'Type the filename of the agent to load:\n'
                filename = input(text)
                fullpath = workingDir + '/' + filename
                try:
                    self.listener.load(fullpath, 0)
                    loaded = True
                    self.isTrained = True
                except:
                    print('Invalid filename')
        elif choice == 3:
            self.paramValues = self.chooseParameters()
            self.listener.startTesting(0, self.paramValues)
            self.episodeNum = 0
            self.dataPoints.clear()
            self.episodeStates.clear()
            while self.checkMessages():
                time.sleep(0.1)
        elif choice == 4:
            text = 'Type the filename to save the agent as:\n'
            workingDir = os.getcwd()
            filename = input(text)
            fullpath = workingDir + '/' + filename
            self.listener.save(fullpath, 0)


    def checkMessages(self):
        while self.listener.getQueue(0).qsize():
            message = self.listener.getQueue(0).get(timeout=0)
            if message.type == Model.Message.EVENT:
                if message.data == Model.Message.EPISODE:
                    self.episodeNum += 1
                    lastEpsilon = self.episodeStates[-1].epsilon
                    totalReward = sum([state.reward for state in self.episodeStates])
                    avgLoss = None if not self.episodeStates[0].loss else sum([state.loss for state in self.episodeStates])/len(self.episodeStates)
                    print('Episode ' + str(self.episodeNum) + ': episilon = ' + str(lastEpsilon) + ', reward = ' + str(totalReward) + ', loss = ' + str(avgLoss))
                    self.dataPoints.append((lastEpsilon, totalReward, avgLoss))
                    self.episodeStates.clear()
                elif message.data == Model.Message.TRAIN_FINISHED:
                    totalReward = sum([reward for _, reward, _ in self.dataPoints])
                    avgReward = totalReward/len(self.dataPoints)
                    print('Total Training Reward: ' + str(totalReward))
                    print('Reward/Episode: ' + str(avgReward))
                    return False
                elif message.data == Model.Message.TEST_FINISHED:
                    totalReward = sum([reward for _, reward, _ in self.dataPoints])
                    avgReward = totalReward / len(self.dataPoints)
                    print('Total Test Reward: ' + str(totalReward))
                    print('Reward/Episode: ' + str(avgReward))
                    return False
            elif message.type == Model.Message.STATE:
                self.episodeStates.append(message.data)
        return True
