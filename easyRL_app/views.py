from django.shortcuts import render
from django.http import HttpResponse
import boto3
from django.conf import settings

session = boto3.session.Session()


# Create your views here.

def index(request):
    # s3 = boto3.client(
    #     's3',
    #     aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    #     aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    #     endpoint_url=settings.AWS_S3_ENDPOINT_URL
    # )

    # for bucket in s3.buckets.all():
    #     print(bucket.name)
    #     # to see items inside buckets
    #     for item in bucket.objects.all():
    #         print(item)   
    my_dict = {"insert_me":"This is the insert from easyRL.views"}
    return render(request, "easyRL_app/index.html", context=my_dict)\

def login(request):
    return render(request, "easyRL_app/login.html", context={})
