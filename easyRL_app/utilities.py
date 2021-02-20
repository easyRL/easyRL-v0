import boto3
import botocore
import uuid

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

def invoke_aws_lambda_func(lambdas, data='{}'):
    # lambdas.list_functions()
    return lambdas.invoke(
        FunctionName='cloudBridge',
        InvocationType='RequestResponse',
        Payload=data,
    )

def generate_jobID():
    return 'jobID'.join(['bingusmingus32', str(uuid.uuid4())])
