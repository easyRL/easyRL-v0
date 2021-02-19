from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseRedirect

from django.shortcuts import redirect, render
from django.shortcuts import render, redirect
from pymemcache.client import base
from . import forms

import boto3
import os

session = boto3.session.Session()

# Create your views here.

def index(request):

        
        my_dict = {}
        files = os.listdir(os.path.join(settings.BASE_DIR, "static/easyRL_app/images"))
        my_dict['files'] = files
        form = forms.HyperParameterForm()
        if request.method == "GET":
            my_dict['form'] = form
            return render(request, "easyRL_app/index.html", context=my_dict)
        
        elif request.method == "POST":
            form = forms.HyperParameterForm(request.POST)
            if form.is_valid():
                print(form.cleaned_data["gamma"])
                print(form.cleaned_data["batch"])
            return render(request, "easyRL_app/index.html", context=my_dict)
 

def login(request):
    form = forms.AwsCredentialForm()
    if request.method == "GET":
        return render(request, "easyRL_app/login.html", context={'form': form})   
    elif request.method == "POST":
        form = forms.AwsCredentialForm(request.POST)
        if form.is_valid():
            form.cleaned_data["aws-hidden"] = 'True'
            request.session['aws-hidden'] = form.cleaned_data["aws-hidden"]
            request.session['secret_key'] = form.cleaned_data["aws_secret_key"]
            request.session['aws-access_key'] = form.cleaned_data["aws_access_key"]
            request.session['token'] = form.cleaned_data["aws_security_token"]
            print("***************************************************")
            print(request.session['aws-hidden'])
        return HttpResponseRedirect("/easyRL_app/")


def get_bucket_contents():
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
    pass

