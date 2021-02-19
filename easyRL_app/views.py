from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse
from django.shortcuts import render
from pymemcache.client import base
from . import forms

import boto3
import os

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
    my_dict = {}

    files = os.listdir(os.path.join(settings.BASE_DIR, "static/easyRL_app/images"))
    my_dict['files'] = files

    return render(request, "easyRL_app/index.html", context=my_dict)

def login(request):
    form = forms.FormName()
    if request.method == "POST":
        form = forms.FormName(request.POST)
        if form.is_valid():
            request.session['secret_key'] = form.cleaned_data["aws_secret_key"]
            request.session['aws-access_key'] = form.cleaned_data["aws_access_key"]
            request.session['token'] = form.cleaned_data["aws_security_token"]
    return render(request, "easyRL_app/login.html", context={'form': form})

