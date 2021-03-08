# This is just to support Azure.
# If you are not deploying there this can be removed.
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
# Define your FaaS Function here.
# Each platform handler will call and pass parameters to this function.
#
# @param request A JSON object provided by the platform handler.
# @param context A platform specific object used to communicate with the cloud platform.
# @returns A JSON object to use as a response.
#

paraMap = {
    '1': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval'],
    '2': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'alpha'],
    '3': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'],
    '4': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'],
    '5': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'alpha'],
}

agentList = [
    {"name": "Deep Q", "index": "1", "supportedEnvs": ["singleDim", "singleDimDescrete"]},
    {"name": "Q Learning", "index": "2", "supportedEnvs": ["singleDimDescrete"]},
    {"name": "DRQN", "index": "3", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},
    {"name": "ADRQN", "index": "4", "supportedEnvs": ["singleDim", "singleDimDescrete", "atari"]},
    {"name": "SARSA", "index": "5", "supportedEnvs": ["singleDimDescrete"]}
]
agentMap = {}
for aa in agentList:
    agentMap[aa['index']] = aa

envList = [
    {"name": "Cart Pole", "index": "1", "type": "singleDim"},
    {"name": "Cart Pole Discrete", "index": "2", "type": "singleDimDescrete"},
    {"name": "Frozen Lake", "index": "3", "type": "singleDimDescrete"},
    {"name": "Pendulum", "index": "4", "type": "singleDim"},
    {"name": "Acrobot", "index": "5", "type": "singleDim"},
    {"name": "Mountain Car", "index": "6", "type": "singleDim"},
    {"name": "Adventure", "index": "7", "type": "atari"},
    {"name": "Air Raid", "index": "8", "type": "atari"},
    {"name": "Alien", "index": "9", "type": "atari"},
    {"name": "Amidar", "index": "10", "type": "atari"},
    {"name": "Assault", "index": "11", "type": "atari"},
    {"name": "Asterix", "index": "12", "type": "atari"},
    {"name": "Asteroids", "index": "13", "type": "atari"},
    {"name": "Atlantis", "index": "14", "type": "atari"},
    {"name": "Bank Heist", "index": "15", "type": "atari"},
    {"name": "Battle Zone", "index": "16", "type": "atari"},
    {"name": "Beam Rider", "index": "17", "type": "atari"},
    {"name": "Berzerk", "index": "18", "type": "atari"},
    {"name": "Bowling", "index": "19", "type": "atari"},
    {"name": "Boxing", "index": "20", "type": "atari"},
    {"name": "Breakout", "index": "21", "type": "atari"},
    {"name": "Carnival", "index": "22", "type": "atari"},
    {"name": "Centipede", "index": "23", "type": "atari"},
    {"name": "Chopper Command", "index": "24", "type": "atari"},
    {"name": "Crazy Climber", "index": "25", "type": "atari"},
    {"name": "Demon Attack", "index": "26", "type": "atari"},
    {"name": "Double Dunk", "index": "27", "type": "atari"},
    {"name": "Elevator Action", "index": "28", "type": "atari"},
    {"name": "Enduro", "index": "29", "type": "atari"},
    {"name": "Fishing Derby", "index": "30", "type": "atari"},
    {"name": "Freeway", "index": "31", "type": "atari"},
    {"name": "Frostbite", "index": "32", "type": "atari"},
    {"name": "Gopher", "index": "33", "type": "atari"},
    {"name": "Gravitar", "index": "34", "type": "atari"},
    {"name": "Hero", "index": "35", "type": "atari"},
    {"name": "Ice Hockey", "index": "36", "type": "atari"},
    {"name": "Jamesbond", "index": "37", "type": "atari"},
    {"name": "Journey Escape", "index": "38", "type": "atari"},
    {"name": "Kangaroo", "index": "39", "type": "atari"},
    {"name": "Krull", "index": "40", "type": "atari"},
    {"name": "Kung Fu Master", "index": "41", "type": "atari"},
    {"name": "Montezuma Revenge", "index": "42", "type": "atari"},
    {"name": "Ms. Pacman", "index": "43", "type": "atari"},
    {"name": "Name this Game", "index": "44", "type": "atari"},
    {"name": "Phoenix", "index": "45", "type": "atari"},
    {"name": "Pitfall", "index": "46", "type": "atari"},
    {"name": "Pong", "index": "47", "type": "atari"},
    {"name": "Pooyan", "index": "48", "type": "atari"},
    {"name": "Private Eye", "index": "49", "type": "atari"},
    {"name": "QBert", "index": "50", "type": "atari"},
    {"name": "River Raid", "index": "51", "type": "atari"},
    {"name": "Road Runner", "index": "52", "type": "atari"},
    {"name": "RoboTank", "index": "53", "type": "atari"},
    {"name": "SeaQuest", "index": "54", "type": "atari"},
    {"name": "Skiing", "index": "55", "type": "atari"},
    {"name": "Solaris", "index": "56", "type": "atari"},
    {"name": "Space Invaders", "index": "57", "type": "atari"},
    {"name": "Star Gunner", "index": "58", "type": "atari"},
    {"name": "Tennis", "index": "59", "type": "atari"},
    {"name": "Time Pilot", "index": "60", "type": "atari"},
    {"name": "Tutankham", "index": "61", "type": "atari"},
    {"name": "Up N Down", "index": "62", "type": "atari"},
    {"name": "Venture", "index": "63", "type": "atari"},
    {"name": "Video Pinball", "index": "64", "type": "atari"},
    {"name": "Wizard of Wor", "index": "65", "type": "atari"},
    {"name": "Yars Revenge", "index": "66", "type": "atari"},
    {"name": "Zaxxon", "index": "67", "type": "atari"}
]
envMap = {}
for ev in envList:
    envMap[ev['index']] = ev

paramConditions = {
    "episodes": {
        "name": "Number of Episodes",
        "min": 1,
        "max": 1000000000,
        "default": 1000,
        "showSlider": False
    },
    "steps": {
        "name": "Max Size",
        "min": 1,
        "max": 1000000000,
        "default": 200,
        "showSlider": False
    },
    "gamma": {
        "name": "Gamma",
        "min": 0,
        "max": 1,
        "default": 0.97,
        "showSlider": True
    },
    "minEpsilon": {
        "name": "Min Epsilon",
        "min": 0,
        "max": 1,
        "default": 0.1,
        "showSlider": True
    },
    "maxEpsilon": {
        "name": "Max Epsilon",
        "min": 0,
        "max": 1,
        "default": 1,
        "showSlider": True
    },
    "decayRate": {
        "name": "Decay Rate",
        "min": 0,
        "max": 0.2,
        "default": 0.018,
        "showSlider": True
    },
    "batchSize": {
        "name": "Batch Size",
        "min": 0,
        "max": 256,
        "default": 32,
        "showSlider": True
    },
    "memorySize": {
        "name": "Memory Size",
        "min": 0,
        "max": 655360,
        "default": 1000,
        "showSlider": False
    },
    "targetInterval": {
        "name": "Target Update Interval",
        "min": 0,
        "max": 100000,
        "default": 200,
        "showSlider": False
    },
    "historyLength": {
        "name": "History Length",
        "min": 0,
        "max": 20,
        "default": 10,
        "showSlider": True
    },
    "alpha": {
        "name": "Alpha",
        "min": 0,
        "max": 1,
        "default": 0.18,
        "showSlider": True
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
        ImageId='ami-01b4fa5b09c9741a8',
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

    continuousTraining = False
    if ("continuousTraining" in arguments):
        continuousTraining = arguments["continuousTraining"]

    modelName = "trainedAgent.bin"
    if continuousTraining:
        modelName = "continuousTraining.bin"

    botoSession = boto3.Session(
        aws_access_key_id=accessKey,
        aws_secret_access_key=secretKey,
        aws_session_token=sessionToken,
        region_name='us-east-1'
    )
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
                        "git clone --branch dataExport https://github.com/RobertCordingly/easyRL-v0")
                    stdout = ssh_stdout.readlines() # DO NOT REMOVE
                    stderr = ssh_stderr.readlines() # DO NOT REMOVE
                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                        "echo " + arguments['instanceType'] + " > tag.txt")
                    inspector.addAttribute("instanceState", "updated")
                else:
                    # Instance type match the tag? If not reboot...
                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                        "cat tag.txt")
                    instanceData = ssh_stdout.readlines()
                    if (arguments['instanceType'] not in instanceData[0]):
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
                                    task = "haltJob"

                            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                                "cat ./data.json")
                            stdout = ssh_stdout.readlines()
                            if (stdout != []):
                                inspector.addAttribute(
                                    "progress", json.loads(stdout[0]))
                            else:
                                inspector.addAttribute("progress", "waiting")
                        else:
                            inspector.addAttribute('instanceState', "idle")

                            if continuousTraining:
                                task = "runJob"
                                inspector.addAttribute('instanceState', "startingJob")
            else:
                inspector.addAttribute('instanceState', "initializing")
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
                            errorMessage += error + " min: " + str(paramConditions[error]['min']) + " max: " + str(paramConditions[error]['max']) + " used: " + str(arguments[error] + " ")
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
                            errorMessage += error + " min: " + str(paramConditions[error]['min']) + " max: " + str(paramConditions[error]['max']) + " used: " + str(arguments[error] + " ")
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
                    command += ' &> /dev/null & sleep 1'

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
