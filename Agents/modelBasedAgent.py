from abc import ABC

from Agents import agent


class ModelBasedAgent(agent.Agent, ABC):
    displayName = 'Model Based Agent'
    newParameters = [agent.Agent.Parameter('Min Epsilon', 0.00, 1.00, 0.01, 0.1, True, True, "The minimum value of epsilon during training; the minimum probability that the model will select a random action over its desired one"),
                     agent.Agent.Parameter('Max Epsilon', 0.00, 1.00, 0.01, 1.0, True, True, "The maximum value of epsilon during training; the maximum/starting probability that the model will select a random action over its desired one"),
                     agent.Agent.Parameter('Decay Rate', 0.00, 0.20, 0.001, 0.018, True, True, "The amount to decrease epsilon by each timestep")]
    parameters = agent.Agent.parameters + newParameters
