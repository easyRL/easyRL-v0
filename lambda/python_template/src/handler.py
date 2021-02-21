# This is just to support Azure.
# If you are not deploying there this can be removed.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import json
import logging
from Inspector import *
import time

import boto3
import paramiko

#
# Define your FaaS Function here.
# Each platform handler will call and pass parameters to this function.
# 
# @param request A JSON object provided by the platform handler.
# @param context A platform specific object used to communicate with the cloud platform.
# @returns A JSON object to use as a response.
#

def listInstances(ec2Client):
    instances = []
    response = ec2Client.describe_instances()
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instances.append(instance)
    return instances

def findOurInstance(ec2Client, jobID):
    instances = listInstances(ec2Client)
    for instance in instances:
        if 'Tags' in instance:
            tags = instance['Tags']
            for keyPair in tags:
                if keyPair['Key'] == 'jobID' and keyPair['Value'] == str(jobID):
                    return instance
    return None

def yourFunction(request, context):
    # Import the module and collect data
    inspector = Inspector()

    accessKey = request['accessKey']
    secretKey = request['secretKey']
    sessionToken = request['sessionToken']
    jobID = request['jobID']
    task = request['task']
    arguments = request['arguments']

    botoSession = boto3.Session (
        aws_access_key_id = accessKey,
        aws_secret_access_key = secretKey,
        aws_session_token = sessionToken, 
        region_name = 'us-east-1'
    )
    
    if (task == "createInstance"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        if (findOurInstance(ec2Client, jobID) is None):

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
                inspector.addAttribute("securityGroupData", str(data))

            except:
                group_name = 'easyrlsecurity'
                response = ec2Client.describe_security_groups(
                    Filters=[
                        dict(Name='group-name', Values=[group_name])
                    ]
                )
                security_group_id = response['SecurityGroups'][0]['GroupId']

            inspector.addAttribute("securityGroupId", str(security_group_id))

            instance = ec2Resource.create_instances(
                ImageId='ami-01b4fa5b09c9741a8',
                MinCount=1,
                MaxCount=1,
                InstanceType=arguments['instanceType'],
                SecurityGroupIds=[security_group_id],
                TagSpecifications = [{
                    "ResourceType": "instance",
                    "Tags": [
                        {
                            'Key': 'jobID',
                            'Value': str(jobID)
                        }
                    ]
                }]
            )
            inspector.addAttribute("instance", str(instance))
        else:
            inspector.addAttribute("error", "Instance already exists.")

    elif (task == "runJob"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')

            command = 'printf "'
            command += str(arguments['environment']) + '\n'
            command += str(arguments['agent']) + '\n'
            command += '1\n'

            paraMap = {
                '1': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval'],
                '2': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'alpha'],
                '3': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'],
                '4': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'batchSize', 'memorySize', 'targetInterval', 'historyLength'],
                '5': ['episodes', 'steps', 'gamma', 'minEpsilon', 'maxEpsilon', 'decayRate', 'alpha'],
            }

            paramList = paraMap[str(arguments['agent'])]
            for param in paramList:
                command += str(arguments[param]) + '\n'

            command += '4\n'
            command += 'trainedAgent.bin\n'
            command += '5\n'
            if (sessionToken != ""):
                command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + secretKey + ' --accessKey ' + accessKey + ' --sessionToken ' + sessionToken + ' --jobID ' + jobID
            else:
                command += '" | python3.7 ./easyRL-v0/EasyRL.py --terminal --secretKey ' + secretKey + ' --accessKey ' + accessKey + ' --jobID ' + jobID
            command += ' &> /dev/null & sleep 1'

            inspector.addAttribute("command", command)

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout=ssh_stdout.readlines()
            inspector.addAttribute("stdout", stdout)
            ssh.close()
        else:
            inspector.addAttribute('error', 'Instance not found.')
    elif (task == "instanceState"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')
        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            if 'State' in ourInstance and 'Name' in ourInstance['State']:
                instanceState = ourInstance['State']
                inspector.addAttribute("instanceState", instanceState['Name'])
            response = ec2Client.describe_instance_status(InstanceIds=[ourInstance['InstanceId']])
            #inspector.addAttribute("response", str(response))
            if 'InstanceStatuses' in response:
                inspector.addAttribute("InstanceStatus", response['InstanceStatuses'][0]['InstanceStatus']['Status'])
                inspector.addAttribute("SystemStatus", response['InstanceStatuses'][0]['SystemStatus']['Status'])
        else:
            inspector.addAttribute("error", "Instance not found.")
    elif (task == "haltJob"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')
            
            command = "pKill python3.7"
            inspector.addAttribute("command", command)

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout=ssh_stdout.readlines()
            inspector.addAttribute("stdout", stdout)
            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")
    elif (task == "isRunning"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')
            
            command = "ps -aux | grep EasyRL.py"
            inspector.addAttribute("command", command)

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout=ssh_stdout.readlines()
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
    elif (task == "pullFile"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')

            filename = arguments['path']
            
            command = "cat " + filename

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout=ssh_stdout.readlines()
            inspector.addAttribute("stdout", stdout)
            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")
    elif (task == "updateEasyRL"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='tcss556', password='secretPassword')
            
            command = "git clone --branch dataExport https://github.com/RobertCordingly/easyRL-v0"

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            stdout=ssh_stdout.readlines()
            inspector.addAttribute("stdout", stdout)
            ssh.close()
        else:
            inspector.addAttribute("error", "Instance not found.")
    elif (task == "terminateInstance"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            instance = ec2Resource.Instance(ourInstance['InstanceId'])
            instance.terminate()
            inspector.addAttribute("message", "Terminated")
        else:
            inspector.addAttribute("error", "Instance not found.")
    elif (task == "createBucket"):
        s3Client = botoSession.client('s3')
        bucketName = 'easyrl-' + str(jobID)
        s3Client.create_bucket(Bucket=bucketName)
    elif (task == "deleteBucket"):
        s3Client = botoSession.client('s3')
        bucketName = 'easyrl-' + str(jobID)
        bucket = s3Client.Bucket(bucketName)
        bucket.objects.all().delete()
        s3Client.delete_bucket(Bucket=bucketName)

    return inspector.finish()