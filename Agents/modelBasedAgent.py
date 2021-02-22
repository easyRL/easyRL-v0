from abc import ABC

from Agents import agent

class ModelBasedAgent(agent.Agent, ABC):
    displayName = 'Model Based Agent'
    newParameters = []
    parameters = agent.Agent.parameters + newParameters
