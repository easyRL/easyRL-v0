from storages.backends.s3boto3 import S3Boto3Storage

class MediaStorage(S3Boto3Storage):
#     location = 'media'
    file_overwrite = True
    default_acl = 'public-read'
#     access_key = 'AKIARNM6CAZHI6J4ZOOD'
#     secret_key = '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'
    region_name = 'us-east-1'
    # bucket_name = 'tcss556.winter2021.varikmp'
#     bucket_name = 'tesst123123'
#     custom_domain = '{}.s3.amazonaws.com'.format(bucket_name)
    url_protocol = "http:"
    # endpoint_url = '{}//{}/{}/'.format(url_protocol, custom_domain, location)
    # endpoint_url = 'http://{}.s3.amazonaws.com/{}'.format(bucket_name, location)
    # endpoint_url = 'http://{}.s3-website.us-east-1.amazonaws.com'.format(bucket_name)
    
    def __init__(self, location, access_key, secret_key, bucket_name):
        self.location = location
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.custom_domain = 'easyrl-{}.s3.amazonaws.com'.format(bucket_name)
