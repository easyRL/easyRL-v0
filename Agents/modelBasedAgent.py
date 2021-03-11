from Agents import agent
from abc import ABC

class ModelBasedAgent(agent.Agent, ABC):
    displayName = 'Model Based Agent'
    newParameters = []
    parameters = agent.Agent.parameters + newParameters
