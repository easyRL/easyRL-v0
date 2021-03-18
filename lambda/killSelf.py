# Kills in current instance.
# By Robert Cordingly

import boto3
import sys

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
        if 'Tags' in instance and 'State' in instance:
            if instance['State']['Name'] != 'pending' and instance['State']['Name'] != 'running':
                continue
            tags = instance['Tags']
            for keyPair in tags:
                if keyPair['Key'] == 'jobID' and keyPair['Value'] == str(jobID):
                    return instance
    return None

def terminateInstance(ec2Client, ec2Resource, ourInstance):
    if (ourInstance is not None):
        instance = ec2Resource.Instance(ourInstance['InstanceId'])
        instance.terminate()

jobID = sys.argv[1]
accessKey = sys.argv[2]
secretKey = sys.argv[3]

if (len(sys.argv) == 5):
	sessionToken = sys.argv[4]
else:
	sessionToken = ""

botoSession = boto3.Session (
	aws_access_key_id = accessKey,
	aws_secret_access_key = secretKey,
	aws_session_token = sessionToken, 
	region_name = 'us-east-1'
)

ec2Client = botoSession.client('ec2')
ec2Resource = botoSession.resource('ec2')
ourInstance = findOurInstance(ec2Client, jobID)
terminateInstance(ec2Client, ec2Resource, ourInstance)