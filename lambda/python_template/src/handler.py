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

        response = ec2Client.create_security_group(
            GroupName='easyrlsecurity',
            Description='EasyRL Security Group',
        )
        security_group_id = response['GroupId']

        inspector.addAttribute("securityGroupId", str(security_group_id))

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

        instance = ec2Resource.create_instances(
            ImageId='ami-01b4fa5b09c9741a8',
            MinCount=1,
            MaxCount=1,
            InstanceType='c4.xlarge',
            SecurityGroupIds=[security_group_id]
        )
        inspector.addAttribute("instance", str(instance))

    elif (task == "runJob"):
        pass
    elif (task == "haltJob"):
        pass
    elif (task == "listInstances"):
        pass
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