import boto3
import botocore
import uuid
from easyRL_app import apps
import os
import json
import core

def get_aws_s3(aws_access_key_id, aws_secret_access_key, aws_session_token=None):
    return boto3.client('s3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )

def get_aws_lambda(aws_access_key_id, aws_secret_access_key, aws_session_token=None, region_name='us-east-1'):
    return boto3.client('lambda',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

def is_valid_aws_credential(aws_access_key_id, aws_secret_access_key, aws_session_token=None):
    try:
        boto3.client('sts',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        ).get_caller_identity()
        return True
    except botocore.exceptions.ClientError:
        return False

def list_items_in_bucket(aws_access_key, aws_secret_key, bucket_name):
    try:
        s3 = get_aws_s3(aws_access_key, aws_secret_key)
        return [item['Key'] for item in s3.list_objects(Bucket=bucket_name)['Contents']]
    except botocore.exceptions.ClientError:
        return None

def download_item_in_bucket(aws_access_key, aws_secret_key, bucket_name, bucket_filename, local_filename):
    try:
        s3 = get_aws_s3(aws_access_key, aws_secret_key)
        s3.download_file(bucket_name, bucket_filename, local_filename)
        return True
    except botocore.exceptions.ClientError:
        return False

# def get_recent_training_data(aws_access_key, aws_secret_key, bucket_name):
#     local_data_file = apps.LOCAL_JSON_FILE.format(bucket_name)
#     download_item_in_bucket(aws_access_key, aws_secret_key, bucket_name, apps.DATA_JSON_FILE, local_data_file)
# 
#     # read the data.json local file the name is changed to JOB_ID.json in /tmp directory
#     file_content = get_file_content_then_delete_file(local_data_file, 'r')
#     
#     # parse the JSON content to JSON object
#     json_content = json.loads(file_content)
#     last_episode = json_content['episodes'][-1]
#     episodeNo = last_episode['episode']
#     
#     # store the values
#     avgLoss = last_episode['avgLoss']
#     avgEpsilon = last_episode['avgEpsilon']
#     totalReward = last_episode['totalReward']
#     avgReward = json_content['avgReward']
# 
#     # read the image data
#     image_file = apps.IMAGE_FILE.format(episodeNo)
#     image_local_file = "{}/static/{}-{}".format(str(core.settings.BASE_DIR), bucket_name, image_file)
#     download_item_in_bucket(aws_access_key, aws_secret_key, bucket_name, image_file, image_local_file)
#     image_data = get_file_content_then_delete_file(image_local_file, 'rb')
#     
#     return avgLoss, avgEpsilon, totalReward, avgReward, image_data

def invoke_aws_lambda_func(lambdas, data='{}'):
    # lambdas.list_functions()
    return lambdas.invoke(
        FunctionName='cloudBridge',
        InvocationType='RequestResponse',
        Payload=data,
    )

def get_file_content_then_delete_file(file_path, option):
    file = open(file_path, option)
    file_content = file.read()
    file.close()
    os.remove(file_path)
    return file_content

def generate_jobID():
    return str(uuid.uuid4())
