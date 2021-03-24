
import paramiko
import boto3
import time
from Inspector import *
import logging
import json
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

#
# Configure options here..
#
awsRegion = 'us-east-1'
backendAMI = 'ami-0bd8cfaa7944aedfe'
githubDefaultRepo = "https://github.com/RobertCordingly/easyRL-v0"
githubDefaultBranch = "dev/rl"

#
# To add a new agent define the information for it here. The index value must corespond to the index used in
# terminal view. After added to agentList, define agent hyper parameter order in paraMap and if there are new
# hyper parameter values add them to paramConditions.
#   
agentList = [
    {"name": "Q Learning", "description": "Basic Q Learning.", "index": "1", "supportedEnvs": ["singleDimDescrete"]},
    {"name": "SARSA", "description": "State Action Reward State Action learning.", "index": "2", "supportedEnvs": ["singleDimDescrete"]},

    {"name": "Deep Q (SRB)", "description": "Deep Q Learning using the standard replay buffer.", "index": "3", "supportedEnvs": ["singleDim", "singleDimDescrete"]},
    {"name": "Deep Q (PRB)", "description": "Deep Q Learning using a prioritized replay buffer.", "index": "4", "supportedEnvs": ["singleDim", "singleDimDescrete"]},
    {"name": "Deep Q (HER)", "description": "Deep Q Learning using a hindsight experience replay buffer.", "index": "5", "supportedEnvs": ["singleDim"]},

    {"name": "DRQN (SRB)", "description": "Deep Recurrent Q-Network using the standard replay buffer.", "index": "6", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},
    {"name": "DRQN (PRB)",  "description": "Deep Recurrent Q-Network using a prioritized replay buffer.", "index": "7", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},
    {"name": "DRQN (HER)",  "description": "Deep Recurrent Q-Network using a hindsight experience replay buffer.", "index": "8", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},

    {"name": "ADRQN (SRB)", "description": "Action-Specific Deep Recurrent Q-Network using the standard replay buffer.", "index": "9", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},
    {"name": "ADRQN (PRB)", "description": "Action-Specific Deep Recurrent Q-Network using the standard replay buffer.", "index": "10", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},
    {"name": "ADRQN (HER)", "description": "Action-Specific Deep Recurrent Q-Network using a hindsight experience replay buffer.", "index": "11", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},

    {"name": "NPG", "description": "Natural Policy Gradient.", "index": "12", "supportedEnvs": ["singleDim"]},
    {"name": "DDPG", "description": "Deep Deterministic Policy Gradient Learning.", "index": "13", "supportedEnvs": ["singleDim"]},

    {"name": "CEM", "description": "Cross Entropy Method Learning.", "index": "14", "supportedEnvs": ["singleDim"]},

    {"name": "SAC", "description": "Soft Actor Critic Learning.", "index": "15", "supportedEnvs": ["singleDim"]},
    {"name": "TRPO", "description": "Trust Region Policy Optimization.", "index": "16", "supportedEnvs": ["singleDim"]},
    {"name": "Rainbow", "description": "Reinforcement learning with the Rainbow agent.", "index": "17", "supportedEnvs": ["singleDim"]}

]

paraMap = {
    '1': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'alpha'], # Q Learning
    '2': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'alpha'], #SARSA

    '3': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval'], # Deep Q
    '4': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'alpha'], # Deep Q
    '5': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval'], # Deep Q

    '6': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'], # DRQN
    '7': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength', 'alpha'], # DRQN
    '8': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'],  # DRQN

    '9': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'], # ADRQN
    '10': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength', 'alpha'],  # ARQN
    '11': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'],  # ADRQN

    '12': ['episodes', 'steps', 'gamma', 'delta'], # NPG
    '13': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'tau'], # DDPG
    '14': ['episodes', 'steps', 'gamma', 'sigma', 'population', 'elite'], # CEM

    '15': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'tau', 'temperature'], # SAC

    '16': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'policyLearnRate', 'valueLearnRate', 'horizon', 'epochSize', 'ppoEpsilon', 'ppoLambda', 'valueLearnRatePlus'], # TRPO
    '17': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'learningRate'] # Rainbow
}

instanceInfo = {
    "c4.large": {
        "cost": 0.1,
        "vcpus": 2,
        "gpus": 0,
        "ram": 3.75,
        "network": "Moderate"
    },
    "c4.xlarge": {
        "cost": 0.19,
        "vcpus": 4,
        "gpus": 0,
        "ram": 7.5,
        "network": "High"
    },
    "c4.2xlarge": {
        "cost": 0.39,
        "vcpus": 8,
        "gpus": 0,
        "ram": 15,
        "network": "High"
    },
    "c4.4xlarge": {
        "cost": 0.79,
        "vcpus": 16,
        "gpus": 0,
        "ram": 30,
        "network": "High"
    },
    "c4.8xlarge": {
        "cost": 1.59,
        "vcpus": 36,
        "gpus": 0,
        "ram": 60,
        "network": "10 Gigabit"
    },
    "g4dn.xlarge": {
        "cost": 0.52,
        "vcpus": 4,
        "gpus": 1,
        "ram": 16,
        "network": "25 Gigabit"
    },
    "g4dn.2xlarge": {
        "cost": 0.75,
        "vcpus": 8,
        "gpus": 1,
        "ram": 32,
        "network": "25 Gigabit"
    },
    "g4dn.4xlarge": {
        "cost": 1.20,
        "vcpus": 16,
        "gpus": 0,
        "ram": 64,
        "network": "25 Gigabit"
    },
    "g4dn.8xlarge": {
        "cost": 2.17,
        "vcpus": 32,
        "gpus": 1,
        "ram": 128,
        "network": "50 Gigabit"
    },
}

agentMap = {}
for aa in agentList:
    agentMap[aa['index']] = aa

envList = [
    {"name": "Cart Pole", "description": "Gain reward by balancing the pole as long as possible.", "index": "1", "type": "singleDim"},
    {"name": "Cart Pole Discrete", "description": "Balance the pole using descrete values instead.",  "index": "2", "type": "singleDimDescrete"},
    {"name": "Frozen Lake", "description": "Navigate the frozen lake without falling in!",  "index": "3", "type": "singleDimDescrete"},
    {"name": "Pendulum", "description": "Swing the pendulum to gain rewards.",  "index": "4", "type": "singleDim"},
    {"name": "Acrobot", "description": "Swing the arm of the robot as high as possible to maximize rewards.",  "index": "5", "type": "singleDim"},
    {"name": "Mountain Car", "description": "Drive the car up the mountains to gain reward.",  "index": "6", "type": "singleDim"},
    {"name": "Adventure", "description": "A classic atari game. Score points to gain rewards.",  "index": "7", "type": "atari"},
    {"name": "Air Raid", "description": "A classic atari game. Score points to gain rewards.",  "index": "8", "type": "atari"},
    {"name": "Alien", "description": "A classic atari game. Score points to gain rewards.",  "index": "9", "type": "atari"},
    {"name": "Amidar", "description": "A classic atari game. Score points to gain rewards.",  "index": "10", "type": "atari"},
    {"name": "Assault", "description": "A classic atari game. Score points to gain rewards.",  "index": "11", "type": "atari"},
    {"name": "Asterix", "description": "A classic atari game. Score points to gain rewards.",  "index": "12", "type": "atari"},
    {"name": "Asteroids", "description": "A classic atari game. Score points to gain rewards.",  "index": "13", "type": "atari"},
    {"name": "Atlantis", "description": "A classic atari game. Score points to gain rewards.",  "index": "14", "type": "atari"},
    {"name": "Bank Heist", "description": "A classic atari game. Score points to gain rewards.",  "index": "15", "type": "atari"},
    {"name": "Battle Zone", "description": "A classic atari game. Score points to gain rewards.",  "index": "16", "type": "atari"},
    {"name": "Beam Rider", "description": "A classic atari game. Score points to gain rewards.",  "index": "17", "type": "atari"},
    {"name": "Berzerk", "description": "A classic atari game. Score points to gain rewards.",  "index": "18", "type": "atari"},
    {"name": "Bowling", "description": "A classic atari game. Score points to gain rewards.",  "index": "19", "type": "atari"},
    {"name": "Boxing", "description": "A classic atari game. Score points to gain rewards.",  "index": "20", "type": "atari"},
    {"name": "Breakout", "description": "A classic atari game. Score points to gain rewards.",  "index": "21", "type": "atari"},
    {"name": "Carnival", "description": "A classic atari game. Score points to gain rewards.",  "index": "22", "type": "atari"},
    {"name": "Centipede", "description": "A classic atari game. Score points to gain rewards.",  "index": "23", "type": "atari"},
    {"name": "Chopper Command", "description": "A classic atari game. Score points to gain rewards.",  "index": "24", "type": "atari"},
    {"name": "Crazy Climber", "description": "A classic atari game. Score points to gain rewards.",  "index": "25", "type": "atari"},
    {"name": "Demon Attack", "description": "A classic atari game. Score points to gain rewards.",  "index": "26", "type": "atari"},
    {"name": "Double Dunk", "description": "A classic atari game. Score points to gain rewards.",  "index": "27", "type": "atari"},
    {"name": "Elevator Action", "description": "A classic atari game. Score points to gain rewards.",  "index": "28", "type": "atari"},
    {"name": "Enduro", "description": "A classic atari game. Score points to gain rewards.",  "index": "29", "type": "atari"},
    {"name": "Fishing Derby", "description": "A classic atari game. Score points to gain rewards.",  "index": "30", "type": "atari"},
    {"name": "Freeway", "description": "A classic atari game. Score points to gain rewards.",  "index": "31", "type": "atari"},
    {"name": "Frostbite", "description": "A classic atari game. Score points to gain rewards.",  "index": "32", "type": "atari"},
    {"name": "Gopher", "description": "A classic atari game. Score points to gain rewards.",  "index": "33", "type": "atari"},
    {"name": "Gravitar", "description": "A classic atari game. Score points to gain rewards.",  "index": "34", "type": "atari"},
    {"name": "Hero", "index": "35", "description": "A classic atari game. Score points to gain rewards.",  "type": "atari"},
    {"name": "Ice Hockey", "description": "A classic atari game. Score points to gain rewards.",  "index": "36", "type": "atari"},
    {"name": "Jamesbond", "description": "A classic atari game. Score points to gain rewards.",  "index": "37", "type": "atari"},
    {"name": "Journey Escape", "description": "A classic atari game. Score points to gain rewards.",  "index": "38", "type": "atari"},
    {"name": "Kangaroo", "description": "A classic atari game. Score points to gain rewards.",  "index": "39", "type": "atari"},
    {"name": "Krull", "description": "A classic atari game. Score points to gain rewards.",  "index": "40", "type": "atari"},
    {"name": "Kung Fu Master", "description": "A classic atari game. Score points to gain rewards.",  "index": "41", "type": "atari"},
    {"name": "Montezuma Revenge", "description": "A classic atari game. Score points to gain rewards.",  "index": "42", "type": "atari"},
    {"name": "Ms. Pacman", "description": "A classic atari game. Score points to gain rewards.",  "index": "43", "type": "atari"},
    {"name": "Name this Game", "description": "A classic atari game. Score points to gain rewards.",  "index": "44", "type": "atari"},
    {"name": "Phoenix", "description": "A classic atari game. Score points to gain rewards.",  "index": "45", "type": "atari"},
    {"name": "Pitfall", "description": "A classic atari game. Score points to gain rewards.",  "index": "46", "type": "atari"},
    {"name": "Pong", "description": "A classic atari game. Score points to gain rewards.",  "index": "47", "type": "atari"},
    {"name": "Pooyan", "description": "A classic atari game. Score points to gain rewards.",  "index": "48", "type": "atari"},
    {"name": "Private Eye", "description": "A classic atari game. Score points to gain rewards.",  "index": "49", "type": "atari"},
    {"name": "QBert", "description": "A classic atari game. Score points to gain rewards.",  "index": "50", "type": "atari"},
    {"name": "River Raid", "description": "A classic atari game. Score points to gain rewards.",  "index": "51", "type": "atari"},
    {"name": "Road Runner", "description": "A classic atari game. Score points to gain rewards.",  "index": "52", "type": "atari"},
    {"name": "RoboTank", "description": "A classic atari game. Score points to gain rewards.",  "index": "53", "type": "atari"},
    {"name": "SeaQuest", "description": "A classic atari game. Score points to gain rewards.",  "index": "54", "type": "atari"},
    {"name": "Skiing", "description": "A classic atari game. Score points to gain rewards.",  "index": "55", "type": "atari"},
    {"name": "Solaris", "description": "A classic atari game. Score points to gain rewards.",  "index": "56", "type": "atari"},
    {"name": "Space Invaders", "description": "A classic atari game. Score points to gain rewards.",  "index": "57", "type": "atari"},
    {"name": "Star Gunner", "description": "A classic atari game. Score points to gain rewards.",  "index": "58", "type": "atari"},
    {"name": "Tennis", "description": "A classic atari game. Score points to gain rewards.",  "index": "59", "type": "atari"},
    {"name": "Time Pilot", "description": "A classic atari game. Score points to gain rewards.",  "index": "60", "type": "atari"},
    {"name": "Tutankham", "description": "A classic atari game. Score points to gain rewards.",  "index": "61", "type": "atari"},
    {"name": "Up N Down", "description": "A classic atari game. Score points to gain rewards.",  "index": "62", "type": "atari"},
    {"name": "Venture", "description": "A classic atari game. Score points to gain rewards.",  "index": "63", "type": "atari"},
    {"name": "Video Pinball", "description": "A classic atari game. Score points to gain rewards.",  "index": "64", "type": "atari"},
    {"name": "Wizard of Wor", "description": "A classic atari game. Score points to gain rewards.",  "index": "65", "type": "atari"},
    {"name": "Yars Revenge", "description": "A classic atari game. Score points to gain rewards.",  "index": "66", "type": "atari"},
    {"name": "Zaxxon", "description": "A classic atari game. Score points to gain rewards.",  "index": "67", "type": "atari"}
]
envMap = {}
for ev in envList:
    envMap[ev['index']] = ev

paramConditions = {
    "episodes": {
        "name": "Number of Episodes",
        "description": "The number of episodes to train the agent.",
        "min": 1,
        "max": 1000000000,
        "default": 1000,
        "showSlider": False
    },
    "steps": {
        "name": "Max Size",
        "description": "The max number of timesteps permitted in an episode.",
        "min": 1,
        "max": 1000000000,
        "default": 200,
        "showSlider": False
    },
    "gamma": {
        "name": "Gamma",
        "description": "The factor by which to discount future rewards.",
        "min": 0,
        "max": 1,
        "default": 0.97,
        "showSlider": True,
        "stepSize": 0.01
    },
    "minEpsilon": {
        "name": "Min Epsilon",
        "description": "The minimum probability that the model will select a random action over its desired one.",
        "min": 0,
        "max": 1,
        "default": 0.1,
        "showSlider": True,
        "stepSize": 0.01
    },
    "maxEpsilon": {
        "name": "Max Epsilon",
        "description": "The maximum/starting probability that the model will select a random action over its desired one.",
        "min": 0,
        "max": 1,
        "default": 1,
        "showSlider": True,
        "stepSize": 0.01
    },
    "decayRate": {
        "name": "Decay Rate",
        "description": "The amount to decrease epsilon by each timestep.",
        "min": 0,
        "max": 0.2,
        "default": 0.018,
        "showSlider": True,
        "stepSize": 0.001
    },
    "batchSize": {
        "name": "Batch Size",
        "description": "The number of transitions to consider simultaneously when updating the agent.",
        "min": 1,
        "max": 256,
        "default": 32,
        "showSlider": True,
        "stepSize": 1
    },
    "memorySize": {
        "name": "Memory Size",
        "description": "The maximum number of timestep transitions to keep stored.",
        "min": 1,
        "max": 655360,
        "default": 1000,
        "showSlider": False
    },
    "targetInterval": {
        "name": "Target Update Interval",
        "description": "The distance in timesteps between target model updates.",
        "min": 1,
        "max": 100000,
        "default": 200,
        "showSlider": False
    },
    "historyLength": {
        "name": "History Length",
        "description": "",
        "min": 0,
        "max": 20,
        "default": 10,
        "showSlider": True,
        "stepSize": 1
    },
    "alpha": {
        "name": "Learning Rate",
        "description": "The rate at which the parameters respond to environment observations.",
        "min": 0,
        "max": 1,
        "default": 0.18,
        "showSlider": True,
        "stepSize": 0.01
    },
    "delta": {
        "name": "Delta",
        "description": "The normalized step size for computing the learning rate.",
        "min": 0,
        "max": 0.05,
        "default": 0.001,
        "showSlider": True,
        "stepSize": 0.0001
    },
    "sigma": {
        "name": "Sigma",
        "description": "The standard deviation of additive noise.",
        "min": 0.001,
        "max": 1,
        "default": 0.5,
        "showSlider": True,
        "stepSize": 0.001
    },
    "population": {
        "name": "Population Size",
        "description": "The size of the sample population.",
        "min": 0,
        "max": 100,
        "default": 10,
        "showSlider": True,
        "stepSize": 1
    },
    "elite": {
        "name": "Elite Fraction",
        "description": "The proportion of the elite to consider for policy improvement.",
        "min": 0.001,
        "max": 1,
        "default": 0.2,
        "showSlider": True,
        "stepSize": 0.001
    },
    "tau": {
        "name": "Tau",
        "description": "",
        "min": 0,
        "max": 1,
        "default": 0.97,
        "showSlider": True,
        "stepSize": 0.001
    },
    "temperature": {
        "name": "Temperature",
        "description": "",
        "min": 0,
        "max": 1,
        "default": 0.97,
        "showSlider": True,
        "stepSize": 0.001
    },
    "learningRate": {
        "name": "Learning Rate",
        "description": "",
        "min": 0.0001,
        "max": 1,
        "default": 0.001,
        "showSlider": True,
        "stepSize": 0.001
    },
    "policyLearnRate": {
        "name": "Policy Learning Rate",
        "description": "",
        "min": 0.0001,
        "max": 1,
        "default": 0.001,
        "showSlider": True,
        "stepSize": 0.001
    },
    "valueLearnRate": {
        "name": "Value Learning Rate",
        "description": "",
        "min": 0.0001,
        "max": 1,
        "default": 0.001,
        "showSlider": True,
        "stepSize": 0.001
    },
    "horizon": {
        "name": "Horizon",
        "description": "",
        "min": 10,
        "max": 10000,
        "default": 50,
        "showSlider": True,
        "stepSize": 0.001
    },
    "epochSize": {
        "name": "Epoch Size",
        "description": "",
        "min": 10,
        "max": 100000,
        "default": 500,
        "showSlider": True,
        "stepSize": 0.001
    },
    "ppoEpsilon": {
        "name": "PPO Epsilon",
        "description": "",
        "min": 0.0001,
        "max": 0.5,
        "default": 0.2,
        "showSlider": True,
        "stepSize": 0.0001
    },
    "ppoLambda": {
        "name": "PPO Lambda",
        "description": "",
        "min": 0.5,
        "max": 1,
        "default": 0.95,
        "showSlider": True,
        "stepSize": 0.01
    },
    "valueLearnRatePlus": {
        "name": "Value Learning Rate+",
        "description": "",
        "min": 0.0001,
        "max": 1,
        "default": 0.001,
        "showSlider": True,
        "stepSize": 0.001
    }
}


def listInstances(ec2Client, inspector):
    instances = []
    response = ec2Client.describe_instances()
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instances.append(instance)
    return instances


def findOurInstance(ec2Client, jobID, inspector):
    instances = listInstances(ec2Client, inspector)
    for instance in instances:
        if 'Tags' in instance and 'State' in instance:
            if instance['State']['Name'] != 'pending' and instance['State']['Name'] != 'running':
                continue
            tags = instance['Tags']
            for keyPair in tags:
                if keyPair['Key'] == 'jobID' and keyPair['Value'] == str(jobID):
                    return instance
    return None


def createInstance(ec2Client, ec2Resource, jobID, arguments, inspector):
    try:
        response = ec2Client.create_security_group(
            GroupName='easyrlsecurity',
            Description='EasyRL Security Group',
        )
        security_group_id = response['GroupId']

        data = ec2Client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {'IpProtocol': 'tcp',
                 'FromPort': 80,
                 'ToPort': 80,
                 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]},
                {'IpProtocol': 'tcp',
                 'FromPort': 22,
                 'ToPort': 22,
                 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}
            ])
        #inspector.addAttribute("securityGroupData", str(data))

    except:
        group_name = 'easyrlsecurity'
        response = ec2Client.describe_security_groups(
            Filters=[
                dict(Name='group-name', Values=[group_name])
            ]
        )
        security_group_id = response['SecurityGroups'][0]['GroupId']

    #inspector.addAttribute("securityGroupId", str(security_group_id))

    instance = ec2Resource.create_instances(
        ImageId=backendAMI,
        MinCount=1,
        MaxCount=1,
        InstanceType=arguments['instanceType'],
        SecurityGroupIds=[security_group_id],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {
                    'Key': 'jobID',
                    'Value': str(jobID)
                }
            ]
        }]
    )
    inspector.addAttribute("message", "created instance")


def terminateInstance(ec2Client, ec2Resource, ourInstance, inspector):
    if (ourInstance is not None):
        instance = ec2Resource.Instance(ourInstance['InstanceId'])
        instance.terminate()
        inspector.addAttribute("message", "terminated instance")
    else:
        inspector.addAttribute("error", "Instance not found.")


def yourFunction(request, context):
    # Import the module and collect data
    inspector = Inspector()

    accessKey = request['accessKey']
    secretKey = request['secretKey']
    sessionToken = request['sessionToken']
    jobID = request['jobID']
    task = request['task']
    arguments = request['arguments']

    if ('episodes' in arguments):
        arguments['episodes'] += 1

    instanceID = ""
    if ('instanceID' in arguments):
        instanceID = arguments['instanceID']
    if (instanceID is not None):
        jobID += str(instanceID)

    if ('gitHubURL' not in arguments):
        arguments['gitHubURL'] = githubDefaultRepo
        arguments['gitHubBranch'] = githubDefaultBranch

    continuousTraining = False
    if ("continuousTraining" in arguments):
        continuousTraining = arguments["continuousTraining"]

    modelName = "model.bin"

    botoSession = boto3.Session(
        aws_access_key_id=accessKey,
        aws_secret_access_key=secretKey,
        aws_session_token=sessionToken,
        region_name=awsRegion
    )
    inspector.addAttribute("instanceStateText", "Loading...")

    if 'instanceType' in arguments:
        try:
            inspector.addAttribute("cost", instanceInfo[arguments['instanceType']]['cost'])
            inspector.addAttribute("info", instanceInfo[arguments['instanceType']])
        except:
            pass

    if (task == "poll"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')
        s3Resource = botoSession.resource('s3')
        try:
            ourInstance = findOurInstance(ec2Client, jobID, inspector)
            inspector.addAttribute("validCredentials", 1)
        except:
            inspector.addAttribute("validCredentials", 0)
            return inspector.finish()
        if (ourInstance is None):
            createInstance(ec2Client, ec2Resource, jobID, arguments, inspector)
            inspector.addAttribute("message", "creating instance")
            inspector.addAttribute("instanceState", "booting")
            inspector.addAttribute("instanceStateText", "Booting")
        else:
            # Check if it is ready to SSH...
            try:
                ip = ourInstance['PublicIpAddress']
                inspector.addAttribute("ip", ip)
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ip, username='tcss556', password='secretPassword')
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                    "echo test")
                stdout = ssh_stdout.readlines()
            except:
                inspector.addAttribute(
                    "error", "Problem creating ssh connection to " + str(ip) + " try again")
                return inspector.finish()

            if (stdout[0] == "test\n"):
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                    "cat easyRL-v0/lambda/version_check1.txt")
                instanceData = ssh_stdout.readlines()
                # Has the version check? If not update
                if (instanceData == []):
                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                        "mv easyRL-v0/ OLD" + str(random.randint(1,10000000)) + "/")
                    stdout = ssh_stdout.readlines()
                    if (sessionToken == ""):
                        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                            "sleep " + str(arguments['killTime']) + " && python3.7 easyRL-v0/lambda/killSelf.py " + jobID + " " + accessKey + " " + secretKey + " &")
                    else:
                        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                            "sleep " + str(arguments['killTime']) + " && python3.7 easyRL-v0/lambda/killSelf.py " + jobID + " " + accessKey + " " + secretKey + " " + sessionToken + " &")
                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                        "git clone --branch " + arguments['gitHubBranch'] + " " + arguments['gitHubURL'])
                    stdout = ssh_stdout.readlines() # DO NOT REMOVE
                    stderr = ssh_stderr.readlines() # DO NOT REMOVE
                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                        "echo " + arguments['instanceType'] + str(arguments['killTime']) + " > tag.txt")
                    inspector.addAttribute("instanceState", "updated")
                    inspector.addAttribute("instanceStateText", "Cloned Repository")
                else:
                    # Instance type match the tag? If not reboot...
                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                        "cat tag.txt")
                    instanceData = ssh_stdout.readlines()
                    tag = arguments['instanceType'] + str(arguments['killTime'])
                    if (instanceData == [] or tag not in instanceData[0]):
                        terminateInstance(
                            ec2Client, ec2Resource, ourInstance, inspector)
                        createInstance(ec2Client, ec2Resource,
                                       jobID, arguments, inspector)
                        try:
                            bucket = s3Resource.Bucket('easyrl-' + jobID)
                            bucket.objects.all().delete()
                        except:
                            pass
                        inspector.addAttribute('instanceState', "rebooting")
                        inspector.addAttribute("instanceStateText", "Recreating")
                    else:
                        # Is job running? If it is get progress. Else return idle.
                        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                            "ps -aux | grep EasyRL.py")
                        stdout = ssh_stdout.readlines()
                        results = ""
                        for line in stdout:
                            results += line
                        if ("terminal" in results):
                            inspector.addAttribute(
                                'instanceState', "runningJob")
                            inspector.addAttribute("instanceStateText", "Running Task")
                            
                            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                                "cat ./arguments.json")
                            stdout = ssh_stdout.readlines()

                            #inspector.addAttribute("Test", str(stdout))
                            #return inspctor.finish()

                            if (stdout != []):
                                jobArguments = json.loads(stdout[0])
                                inspector.addAttribute(
                                    "jobArguments", jobArguments)
                            
                                if continuousTraining and jobArguments != arguments:
                                    inspector.addAttribute('instanceState', "changingJob")
                                    inspector.addAttribute("instanceStateText", "Changing Task")
                                    task = "haltJob"

                            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                                "cat ./data.json")
                            stdout = ssh_stdout.readlines()
                            if (stdout != []):
                                try:
                                    inspector.addAttribute(
                                        "progress", json.loads(stdout[0]))
                                except:
                                    inspector.addAttribute("progress", "waiting")
                            else:
                                inspector.addAttribute("progress", "waiting")
                        else:
                            inspector.addAttribute('instanceState', "idle")
                            inspector.addAttribute("instanceStateText", "Idle")

                            if continuousTraining:
                                task = "runJob"
                                inspector.addAttribute('instanceState', "startingJob")
                                inspector.addAttribute("instanceStateText", "Starting Task")
            else:
                inspector.addAttribute('instanceState', "initializing")
                inspector.addAttribute("instanceStateText", "Initializing")
            ssh.close()

    if (task == "runJob"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')
        s3Resource = botoSession.resource('s3')
        try:
            bucket = s3Resource.Bucket('easyrl-' + jobID)
            bucket.objects.all().delete()
        except:
            pass

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ip, username='tcss556', password='secretPassword')

                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                                "ps -aux | grep EasyRL.py")
                stdout = ssh_stdout.readlines()

            except:
                inspector.addAttribute(
                    "error", "Problem creating ssh connection to " + str(ip) + " try again")
                return inspector.finish()

            results = ""
            for line in stdout:
                results += line
            if ("terminal" not in results):

                # Error Checking
                if (str(arguments['agent']) in paraMap):
                    missingAttributes = []
                    outOfRange = []
                    valid = True
                    envIndex = str(arguments['environment'])
                    agentIndex = str(arguments['agent'])
                    if envMap[envIndex]['type'] not in agentMap[agentIndex]['supportedEnvs']:
                        inspector.addAttribute("error", "Incompatible agent/environment pair!")
                        return inspector.finish()
                    for pp in paraMap[str(arguments['agent'])]:
                        pp = str(pp)
                        if pp not in arguments:
                            missingAttributes.append(pp)
                        else:
                            val = arguments[pp]
                            if (val < paramConditions[pp]['min'] or val > paramConditions[pp]['max']):
                                outOfRange.append(pp)
                    if len(missingAttributes) > 0:
                        inspector.addAttribute("error-Missing", "Missing hyperparameters for agent: " + str(missingAttributes))
                        valid = False
                    if len(outOfRange) > 0:
                        errorMessage = "Attributes with invalid value: "
                        for error in outOfRange:
                            errorMessage += error + " min: " + str(paramConditions[error]['min']) + " max: " + str(paramConditions[error]['max']) + " used: " + str(arguments[error]) + " "
                        inspector.addAttribute("error-Range", errorMessage)
                        valid = False
                    if (valid == False):
                        return inspector.finish()
                else:
                    inspector.addAttribute("error", "Unknown Agent " + str(arguments['agent']))
                    return inspector.finish()


                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                                "echo \'" + json.dumps(arguments) + "\' > arguments.json")
                stdout = ssh_stdout.readlines()

                command = 'printf "'
                command += str(arguments['environment']) + '\n'
                command += str(arguments['agent']) + '\n'
                command += '1\n'
                paramList = paraMap[str(arguments['agent'])]
                for param in paramList:
                    command += str(arguments[param]) + '\n'

                command += '4\n'
                command += modelName + '\n'
                command += '5\n'
                if (sessionToken != ""):
                    command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + secretKey + \
                        ' --accessKey ' + accessKey + ' --sessionToken ' + \
                        sessionToken + ' --jobID ' + jobID
                else:
                    command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + \
                        secretKey + ' --accessKey ' + accessKey + ' --jobID ' + jobID
                command += ' &> lastJobLog.txt & sleep 1'

                #inspector.addAttribute("command", command)

                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
                stdout = ssh_stdout.readlines()
                #inspector.addAttribute("stdout", stdout)
                ssh.close()
                inspector.addAttribute("message", "Job started")
            else:
                inspector.addAttribute("message", "Job already running")
        else:
            inspector.addAttribute('error', 'Instance not found.')
    
    if (task == "runTest"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')
        s3Resource = botoSession.resource('s3')
        try:
            bucket = s3Resource.Bucket('easyrl-' + jobID)
            bucket.objects.all().delete()
        except:
            pass

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ip, username='tcss556', password='secretPassword')

                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                                "ps -aux | grep EasyRL.py")
                stdout = ssh_stdout.readlines()

            except:
                inspector.addAttribute(
                    "error", "Problem creating ssh connection to " + str(ip) + " try again")
                return inspector.finish()

            results = ""
            for line in stdout:
                results += line
            if ("terminal" not in results):

                # Error Checking
                if (str(arguments['agent']) in paraMap):
                    missingAttributes = []
                    outOfRange = []
                    valid = True
                    envIndex = str(arguments['environment'])
                    agentIndex = str(arguments['agent'])
                    if envMap[envIndex]['type'] not in agentMap[agentIndex]['supportedEnvs']:
                        inspector.addAttribute("error", "Incompatible agent/environment pair!")
                        return inspector.finish()
                    for pp in paraMap[str(arguments['agent'])]:
                        pp = str(pp)
                        if pp not in arguments:
                            missingAttributes.append(pp)
                        else:
                            val = arguments[pp]
                            if (val < paramConditions[pp]['min'] or val > paramConditions[pp]['max']):
                                outOfRange.append(pp)
                    if len(missingAttributes) > 0:
                        inspector.addAttribute("error-Missing", "Missing hyperparameters for agent: " + str(missingAttributes))
                        valid = False
                    if len(outOfRange) > 0:
                        errorMessage = "Attributes with invalid value: "
                        for error in outOfRange:
                            errorMessage += error + " min: " + str(paramConditions[error]['min']) + " max: " + str(paramConditions[error]['max']) + " used: " + str(arguments[error]) + " "
                        inspector.addAttribute("error-Range", errorMessage)
                        valid = False
                    if (valid == False):
                        return inspector.finish()
                else:
                    inspector.addAttribute("error", "Unknown Agent " + str(arguments['agent']))
                    return inspector.finish()


                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                                "echo \'" + json.dumps(arguments) + "\' > arguments.json")
                stdout = ssh_stdout.readlines()

                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                    "md5sum " + modelName)
                instanceData = ssh_stdout.readlines()
                # Has the tag? If not update
                if (instanceData != []):

                    command = 'printf "'
                    command += str(arguments['environment']) + '\n'
                    command += str(arguments['agent']) + '\n'
                    command += '2\n'
                    command += modelName + '\n'
                    command += '3\n'

                    paramList = paraMap[str(arguments['agent'])]
                    for param in paramList:
                        command += str(arguments[param]) + '\n'

                    command += '5\n'
                    if (sessionToken != ""):
                        command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + secretKey + \
                            ' --accessKey ' + accessKey + ' --sessionToken ' + \
                            sessionToken + ' --jobID ' + jobID
                    else:
                        command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + \
                            secretKey + ' --accessKey ' + accessKey + ' --jobID ' + jobID
                    command += ' &> lastJobLog.txt & sleep 1'

                    #inspector.addAttribute("command", command)

                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
                    stdout = ssh_stdout.readlines()
                    #inspector.addAttribute("stdout", stdout)
                    ssh.close()
                    inspector.addAttribute("message", "Test started")
                else:
                    ssh.close()
                    inspector.addAttribute("error", "No trained agent found")
            else:
                inspector.addAttribute("message", "Test already running")
        else:
            inspector.addAttribute('error', 'Instance not found.')

    if (task == "haltJob"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            #inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')

            command = "pkill python3.7"
            #inspector.addAttribute("command", command)

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout = ssh_stdout.readlines()
            #inspector.addAttribute("stdout", stdout)
            ssh.close()
            inspector.addAttribute("message", "Job halted.")
        else:
            inspector.addAttribute("error", "Instance not found.")
    
    if (task == "exportModel"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            #inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                    "md5sum " + modelName)
            instanceData = ssh_stdout.readlines()
            # Has the tag? If not update
            if (instanceData != []):
                if (sessionToken == ""):
                    command = "python3.7 easyRL-v0/lambda/upload.py " + modelName + " " + jobID + " " + accessKey + " " + secretKey 
                else:
                    command = "python3.7 easyRL-v0/lambda/upload.py " + modelName + " " + jobID + " " + accessKey + " " + secretKey + " " + sessionToken 

                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
                stdout = ssh_stdout.readlines()
                inspector.addAttribute("url", "https://easyrl-" + str(jobID) + ".s3.amazonaws.com/" + modelName)
            else:
                inspector.addAttribute("error", "Model not trained yet!")
            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")

    if (task == "import"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')
     
            if (sessionToken == ""):
                command = "python3.7 easyRL-v0/lambda/download.py " + modelName + " " + jobID + " " + accessKey + " " + secretKey 
            else:
                command = "python3.7 easyRL-v0/lambda/download.py " + modelName + " " + jobID + " " + accessKey + " " + secretKey + " " + sessionToken 

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout = ssh_stdout.readlines()
            inspector.addAttribute("error", stdout)

            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")


    if (task == "jobLog"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            #inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')

            if (sessionToken == ""):
                command = "python3.7 easyRL-v0/lambda/upload.py lastJobLog.txt " + jobID + " " + accessKey + " " + secretKey 
            else:
                command = "python3.7 easyRL-v0/lambda/upload.py lastJobLog.txt " + jobID + " " + accessKey + " " + secretKey + " " + sessionToken 

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout = ssh_stdout.readlines()
            inspector.addAttribute("url", "https://easyrl-" + str(jobID) + ".s3.amazonaws.com/lastJobLog.txt")
            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")

    if (task == "terminateInstance"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')
        s3Resource = botoSession.resource('s3')
        try:
            bucket = s3Resource.Bucket('easyrl-' + jobID)
            bucket.objects.all().delete()
        except:
            pass

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        terminateInstance(ec2Client, ec2Resource, ourInstance, inspector)

    if (task == "info"):
        inspector.addAttribute("environments", envList)
        inspector.addAttribute("environmentsMap", envMap)
        inspector.addAttribute("parameters", paramConditions)

        combinedAgents = []
        for agent in agentList:
            agent['parameters'] = paraMap[agent['index']]
            combinedAgents.append(agent)

        combinedAgentsMap = {}
        for aa in combinedAgents:
            combinedAgentsMap[aa['index']] = aa

        inspector.addAttribute("agents", combinedAgents)
        inspector.addAttribute("agentsMap", combinedAgentsMap)

    return inspector.finish()
