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

paramConditions = {
    "episodes": {
        "min": 1,
        "max": 1000000000
    },
    "steps": {
        "min": 1,
        "max": 1000000000
    },
    "gamma": {
        "min": 0,
        "max": 1
    },
    "minEpsilon": {
        "min": 0,
        "max": 1
    },
    "maxEpsilon": {
        "min": 0,
        "max": 1
    },
    "decayRate": {
        "min": 0,
        "max": 0.2
    },
    "batchSize": {
        "min": 0,
        "max": 256
    },
    "memorySize": {
        "min": 0,
        "max": 655360
    },
    "targetInterval": {
        "min": 0,
        "max": 100000
    },
    "historyLength": {
        "min": 0,
        "max": 20
    },
    "alpha": {
        "min": 0,
        "max": 1
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
                    "cat tag.txt")
                instanceData = ssh_stdout.readlines()
                # Has the tag? If not update
                if (instanceData == []):
                    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
                        "mv easyRL-v0/ OLD/")
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
                                inspector.addAttribute(
                                    "jobArguments", json.loads(stdout[0]))

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
            else:
                inspector.addAttribute('instanceState', "initializing")
            ssh.close()

    elif (task == "runJob"):
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
                command += 'trainedAgent.bin\n'
                command += '5\n'
                if (sessionToken != ""):
                    command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + secretKey + \
                        ' --accessKey ' + accessKey + ' --sessionToken ' + \
                        sessionToken + ' --jobID ' + jobID
                else:
                    command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + \
                        secretKey + ' --accessKey ' + accessKey + ' --jobID ' + jobID
                command += ' &> /dev/null & sleep 1'

                inspector.addAttribute("command", command)

                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
                stdout = ssh_stdout.readlines()
                #inspector.addAttribute("stdout", stdout)
                ssh.close()
                inspector.addAttribute("message", "Job started")
            else:
                inspector.addAttribute("message", "Job already running")
        else:
            inspector.addAttribute('error', 'Instance not found.')
    elif (task == "runTest"):
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
                command += '2\n'
                command += 'trainedAgent.bin\n'
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

                inspector.addAttribute("command", command)

                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
                stdout = ssh_stdout.readlines()
                #inspector.addAttribute("stdout", stdout)
                ssh.close()
                inspector.addAttribute("message", "Test started")
            else:
                inspector.addAttribute("message", "Test already running")
        else:
            inspector.addAttribute('error', 'Instance not found.')

    elif (task == "haltJob"):
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
        else:
            inspector.addAttribute("error", "Instance not found.")
    
    elif (task == "exportModel"):
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
                command = "python3.7 easyRL-v0/lambda/upload.py trainedAgent.bin " + jobID + " " + accessKey + " " + secretKey 
            else:
                command = "python3.7 easyRL-v0/lambda/upload.py trainedAgent.bin " + jobID + " " + accessKey + " " + secretKey + " " + sessionToken 

            #inspector.addAttribute("command", command)

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout = ssh_stdout.readlines()
            inspector.addAttribute("url", "https://easyrl-" + str(jobID) + ".s3.amazonaws.com/trainedAgent.bin")
            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")

    elif (task == "terminateInstance"):
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


        """
            Deprecated PROBABLY WILL BE REMOVED. 
            Use poll instead for the vast majority of these functions.
        """

    elif (task == "isRunning"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')

            command = "ps -aux | grep EasyRL.py"
            inspector.addAttribute("command", command)

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout = ssh_stdout.readlines()
            inspector.addAttribute("stdout", stdout)

            results = ""
            for line in stdout:
                results += line

            if ("terminal" in results):
                inspector.addAttribute("isRunning", 1)
            else:
                inspector.addAttribute("isRunning", 0)
            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")


    elif (task == "instanceState"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')
        ourInstance = findOurInstance(ec2Client, jobID, inspector)
        if (ourInstance is not None):
            if 'State' in ourInstance and 'Name' in ourInstance['State']:
                instanceState = ourInstance['State']
                inspector.addAttribute("instanceState", instanceState['Name'])
            response = ec2Client.describe_instance_status(
                InstanceIds=[ourInstance['InstanceId']])
            #inspector.addAttribute("response", str(response))
            if 'InstanceStatuses' in response:
                inspector.addAttribute(
                    "InstanceStatus", response['InstanceStatuses'][0]['InstanceStatus']['Status'])
                inspector.addAttribute(
                    "SystemStatus", response['InstanceStatuses'][0]['SystemStatus']['Status'])
        else:
            inspector.addAttribute("error", "Instance not found.")
            
    elif (task == "isReady"):
        try:
            ec2Client = botoSession.client('ec2')
            ec2Resource = botoSession.resource('ec2')

            ourInstance = findOurInstance(ec2Client, jobID, inspector)
            if (ourInstance is not None):
                ip = ourInstance['PublicIpAddress']
                inspector.addAttribute("ip", str(ip))

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ip, username='tcss556', password='secretPassword')

                command = "echo test"

                #inspector.addAttribute("command", command)
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
                stdout = ssh_stdout.readlines()
                #inspector.addAttribute("stdout", stdout)
                ssh.close()

                if (stdout[0] == "test\n"):
                    inspector.addAttribute('isReady', 1)
                else:
                    inspector.addAttribute('isReady', 0)

            else:
                inspector.addAttribute('error', 'Instance not found.')
                inspector.addAttribute('isReady', 0)
        except:
            inspector.addAttribute('isReady', 0)

    elif (task == "createInstance"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        if (findOurInstance(ec2Client, jobID, inspector) is None):
            createInstance(ec2Client, ec2Resource, jobID, arguments, inspector)
        else:
            inspector.addAttribute("error", "Instance already exists.")

    return inspector.finish()
