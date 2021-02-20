import boto3

def get_aws_s3(aws_access_key_id, aws_secret_access_key):
    return boto3.client('s3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

def get_aws_lambda(aws_access_key_id, aws_secret_access_key, region_name='us-east-1'):
    return boto3.client('lambda',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

def invoke_aws_lambda_func(lambdas, data='{}'):
    # lambdas.list_functions()
    return lambdas.invoke(
        FunctionName='cloudBridge',
        InvocationType='RequestResponse',
        Payload=data,
    )
