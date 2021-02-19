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
from paramiko import SSHClient

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
                InstanceType='c4.xlarge',
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
            inspector.addAttribute("error", "Instance already exists")

    elif (task == "runJob"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            ip = ourInstance['PublicIpAddress']
            inspector.addAttribute("ip", str(ip))

            ssh = paramiko.SSHClient()
            ssh.connect(ip, username='tcss556', password='secretPassword')

            command = 'printf "'
            command += str(arguments['environment']) + '\n'
            command += str(arguments['agent']) + '\n'
            command += '1\n'
            command += str(arguments['episodes']) + '\n'
            command += str(arguments['steps']) + '\n'
            command += str(arguments['gamma']) + '\n'
            command += str(arguments['minEpsilon']) + '\n'
            command += str(arguments['maxEpsilon']) + '\n'
            command += str(arguments['decayRate']) + '\n'
            command += str(arguments['batchSize']) + '\n'
            command += str(arguments['memorySize']) + '\n'
            command += str(arguments['targetInterval']) + '\n'
            command += '4\n'
            command += 'trainedAgent.bin\n'
            command += '5\n'
            command += '" | python3.7 EasyRL.py --terminal --secretKey ' + secretKey + ' --accessKey ' + accessKey + ' --sessionToken ' + sessionToken + ' --jobID ' + jobID
            command += '&> /dev/null & sleep 2'

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)

            ssh.close()

            pass

    elif (task == "haltJob"):
        pass
    elif (task == "listInstances"):
        pass
    elif (task == "terminateInstance"):
        ec2Client = botoSession.client('ec2')
        ec2Resource = botoSession.resource('ec2')

        ourInstance = findOurInstance(ec2Client, jobID)
        if (ourInstance is not None):
            instance = ec2Resource.Instance(ourInstance['InstanceId'])
            instance.terminate()
            inspector.addAttribute("message", "Terminated")
        else:
            inspector.addAttribute("message", "Instance not found.")
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