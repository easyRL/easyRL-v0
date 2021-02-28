import random
import numpy as np
from Agents import drqn, modelBasedAgent, modelFreeAgent
import cProfile

class Model:
    def __init__(self):
        # these can be set directly from the Controller based on user input from the View
        self.environment_class = None
        self.agent_class = None
        self.isHalted = False
        self.isRunning = False
        self.environment = None
        self.agent = None
        self.loadFilename = None

    # def run_learning(self, messageQueue, total_episodes, max_steps, *model_args):
    #     cProfile.runctx('self.run_learning2(messageQueue, total_episodes, max_steps, *model_args)', globals(), locals(),
    #                     'stats')

    # def run_learning2(self, messageQueue, total_episodes, max_steps, *model_args):
    def run_learning(self, messageQueue, total_episodes, max_steps, *model_args):
        self.isRunning = True

        if not self.environment:
            self.environment = self.environment_class()

        if self.loadFilename:
            self.agent = self.agent_class(self.environment.state_size, self.environment.action_size, *model_args)
            self.agent.load(self.loadFilename)
            self.loadFilename = None
        elif not self.agent:
            self.agent = self.agent_class(self.environment.state_size, self.environment.action_size, *model_args)
        else:  # if agent already exists, update the model arguments
            mem = self.agent.memsave()
            self.agent = self.agent_class(self.environment.state_size, self.environment.action_size, *model_args)
            self.agent.memload(mem)
        
        if (isinstance(self.agent, modelFreeAgent.ModelFreeAgent)):
            '''
            Training algorithm for Model Free Agents.
            '''
            min_epsilon, max_epsilon, decay_rate = self.agent.min_epsilon, self.agent.max_epsilon, self.agent.decay_rate
            epsilon = max_epsilon

            for episode in range(int(total_episodes)):
                self.environment.reset()

                for step in range(int(max_steps)):
                    old_state = self.environment.state
                    exp_exp_tradeoff = random.uniform(0, 1)

                    if exp_exp_tradeoff > epsilon:
                        action = self.agent.choose_action(old_state)
                    else:
                        action = self.environment.sample_action()

                    reward = self.environment.step(action)

                    loss = self.agent.remember(old_state, action, reward, self.environment.state, self.environment.done)

                    frame = self.environment.render()
                    modelState = Model.State(frame, epsilon, reward, loss)

                    message = Model.Message(Model.Message.STATE, modelState)
                    messageQueue.put(message)

                    if self.environment.done or self.isHalted:
                        break

                message = Model.Message(Model.Message.EVENT, Model.Message.EPISODE)
                messageQueue.put(message)

                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

                if self.isHalted:
                    self.isHalted = False
                    break
        elif (isinstance(self.agent, modelBasedAgent.ModelBasedAgent)):
            '''
            Training algorithm for Model Based Agents.
            '''
            for episode in range(int(total_episodes)):
                # Reset the environment.
                state = self.environment.reset()
                
                
                # Evaluate the policy
                # Array for the rewards for each policy
                episode_rewards = np.array([])
                for policy in self.agent.get_policies():
                    # Set the policy to the current on.
                    self.agent.set_policy(policy)
                    # Reset the environment.
                    state = self.environment.reset()
                    # Sum of total policy rewards for this episode.
                    policy_reward = 0.0
                    # Execute this episode for each policy.
                    for step in range(int(max_steps)):
                        # Execute one step.
                        old_state = self.environment.state
                        action = self.agent.choose_action(old_state)
                        reward = self.environment.step(action)
                        
                        # Add the reward to the total policy reward
                        policy_reward += reward
                        
                        # Render and save the step.
                        frame = self.environment.render()
                        
                        # Send the state from the step.
                        modelState = Model.State(frame, None, reward, None)
                        message = Model.Message(Model.Message.STATE, modelState)
                        messageQueue.put(message)

                        if self.environment.done or self.isHalted:
                            break
                    
                    # Add the policy rewards to the episode rewards.
                    episode_rewards = np.append(episode_rewards, policy_reward)
                    
                    if self.isHalted:
                        break
                    
                    
                
                # Improve the Policy
                self.agent.update(episode_rewards)
                
                
                
                message = Model.Message(Model.Message.EVENT, Model.Message.EPISODE)
                messageQueue.put(message)

                if self.isHalted:
                    self.isHalted = False
                    break

        message = Model.Message(Model.Message.EVENT, Model.Message.TRAIN_FINISHED)
        messageQueue.put(message)
        self.isRunning = False
        print('learning done')

    def run_testing(self, messageQueue, total_episodes, max_steps, *model_args):
        total_episodes = int(total_episodes+0.5)
        max_steps = int(max_steps+0.5)
        self.isRunning = True

        if not self.environment:
            self.environment = self.environment_class()

        if self.loadFilename:
            self.agent = self.agent_class(self.environment.state_size, self.environment.action_size, *model_args)
            self.agent.load(self.loadFilename)
            self.loadFilename = None
        elif not self.agent:
            return

        if self.agent:
            if (isinstance(self.agent, modelFreeAgent.ModelFreeAgent)):
                '''
                Testing algorithm for Model Free Agents.
                '''
                min_epsilon, max_epsilon, decay_rate = self.agent.min_epsilon, self.agent.max_epsilon, self.agent.decay_rate
                epsilon = max_epsilon

                for episode in range(int(total_episodes)):
                    self.environment.reset()

                    for step in range(int(max_steps)):
                        old_state = self.environment.state

                        exp_exp_tradeoff = random.uniform(0, 1)

                        if exp_exp_tradeoff > epsilon:
                            action = self.agent.choose_action(old_state)
                        else:
                            action = self.environment.sample_action()

                        reward = self.environment.step(action)

                        if isinstance(self.agent, drqn.DRQN):
                            self.agent.addToMemory(old_state, action, reward, self.environment.state, episode, self.environment.done)

                        frame = self.environment.render()
                        
                        # Send the state from the step.
                        modelState = Model.State(frame, None, reward, None)
                        message = Model.Message(Model.Message.STATE, modelState)
                        messageQueue.put(message)

                        if self.environment.done or self.isHalted:
                            break

                    message = Model.Message(Model.Message.EVENT, Model.Message.EPISODE)
                    messageQueue.put(message)

                    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

                    if self.isHalted:
                        self.isHalted = False
                        break
            elif (isinstance(self.agent, modelBasedAgent.ModelBasedAgent)):
                '''
                Testing algorithm for Model Based Agents.
                '''
                for episode in range(int(total_episodes)):
                    # Reset the environment.
                    self.environment.reset()

                    # Execute this episode.
                    for step in range(int(max_steps)):
                        # Execute one step.
                        old_state = self.environment.state
                        action = self.agent.choose_action(old_state)
                        reward = self.environment.step(action)
                        
                        # Render the step
                        frame = self.environment.render()
                        
                        modelState = Model.State(frame, None, reward, None)
                        message = Model.Message(Model.Message.STATE, modelState)
                        messageQueue.put(message)

                        if self.environment.done or self.isHalted:
                            break

                    message = Model.Message(Model.Message.EVENT, Model.Message.EPISODE)
                    messageQueue.put(message)

                    if self.isHalted:
                        self.isHalted = False
                        break
            message = Model.Message(Model.Message.EVENT, Model.Message.TEST_FINISHED)
            messageQueue.put(message)
            print('testing done')
        self.isRunning = False

    def halt_learning(self):
        if self.isRunning:
            self.isHalted = True

    def reset(self):
        self.environment = None
        self.agent = None

    def save(self, filename):
        if self.agent:
            self.agent.save(filename)

    def load(self, filename):
        self.loadFilename = filename

    class Message:
        # types of message
        STATE = 0
        EVENT = 1

        # event types
        TRAIN_FINISHED = 0
        TEST_FINISHED = 1
        EPISODE = 2

        def __init__(self, type, data):
            self.type = type
            self.data = data

    class State:
        def __init__(self, image, epsilon, reward, loss):
            self.image = image
            self.epsilon = epsilon
            self.reward = reward
            self.loss = loss
