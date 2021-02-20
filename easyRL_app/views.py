from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseRedirect

from django.shortcuts import redirect, render
from django.shortcuts import render, redirect
from pymemcache.client import base
from . import forms

import boto3
import os
from easyRL_app.utilities import get_aws_s3, get_aws_lambda,\
    invoke_aws_lambda_func, is_valid_aws_credential, generate_jobID

DEBUG_JOB_ID = generate_jobID()

session = boto3.session.Session()

# Create your views here.

def index(request):
    # send the user back to the login form if the user did not sign in or session expired
    debug_sessions(request)
    if 'aws_succeed' not in request.session :#or not request.session['aws_succeed']:
        #return HttpResponseRedirect("/easyRL_app/login/")
        pass

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
            my_dict['form'] = form
        return render(request, "easyRL_app/index.html", context=my_dict)

def login(request):
    form = forms.AwsCredentialForm()
    if request.method == "GET":
        return render(request, "easyRL_app/login.html", context={'form': form})
    elif request.method == "POST":
        form = forms.AwsCredentialForm(request.POST)
        if form.is_valid() and is_valid_aws_credential(
            form.cleaned_data["aws_access_key"], 
            form.cleaned_data["aws_secret_key"], 
            form.cleaned_data["aws_security_token"]):
            request.session['aws_access_key'] = form.cleaned_data["aws_access_key"]
            request.session['aws_secret_key'] = form.cleaned_data["aws_secret_key"]
            request.session['aws_security_token'] = form.cleaned_data["aws_security_token"]
            request.session['aws_succeed'] = True
            request.session['job_id'] = generate_jobID()
            # create ec2 instance
            debug_sessions(request)
            lambda_create_instance(
                request.session['aws_access_key'],
                request.session['aws_secret_key'],
                request.session['aws_security_token'],
                request.session['job_id']
            )
            return HttpResponseRedirect("/easyRL_app/")
        else:
            request.session['aws_succeed'] = False
            return HttpResponseRedirect("/easyRL_app/login/")

def logout(request):
    # store the keys (to avoid deep copy)
    keys = [key for key in request.session.keys()]
    # terminate the instance for the user
    lambda_terminate_instance(
        request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id']
    )
    # clear up all sessions
    for key in keys:
        del request.session[key]
    return HttpResponseRedirect("/easyRL_app/login/")

def lambda_create_instance(aws_access_key, aws_secret_key, aws_security_token, job_id):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": "createInstance",
        "arguments": {"instanceType": "c4.xlarge"},
    }
    invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))

def lambda_terminate_instance(aws_access_key, aws_secret_key, aws_security_token, job_id):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": "terminateInstance",
        "arguments": "",
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    return HttpResponse(str(response))

def test_data(request):
    if request.method == 'GET' and 'name' in request.GET:
        return HttpResponse("Hello {}".format(request.GET['name']))
    return HttpResponse("Hello World")

def test_create_instance(request):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": os.getenv("AWS_ACCESS_KEY_ID"),
        "secretKey": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "sessionToken": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "jobID": DEBUG_JOB_ID,
        "task": "createInstance",
        "arguments": {"instanceType": "c4.xlarge"},
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    return HttpResponse(str(response))

def test_terminate_instance(request):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": os.getenv("AWS_ACCESS_KEY_ID"),
        "secretKey": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "sessionToken": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "jobID": DEBUG_JOB_ID,
        "task": "terminateInstance",
        "arguments": "",
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    return HttpResponse(str(response))

def debug_sessions(request):
    for key in request.session.keys():
        print("{}={}".format(key, request.session[key]))

