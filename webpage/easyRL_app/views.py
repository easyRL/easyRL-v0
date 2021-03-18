from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt
from . import forms
import time
import json
import boto3
import os
from easyRL_app.utilities import get_aws_lambda,\
    invoke_aws_lambda_func, is_valid_aws_credential, generate_jobID
from easyRL_app import apps

DEBUG_JOB_ID = generate_jobID()
session = boto3.session.Session()

# Create your views here.
def index(request):
    # send the user back to the login form if the user did not sign in or session expired
    debug_sessions(request)
    if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
        return HttpResponseRedirect("/easyRL_app/login/")

    index_dict = {}
    files = os.listdir(os.path.join(settings.BASE_DIR, "static/easyRL_app/images"))
    index_dict['files'] = files
    form = forms.HyperParameterFormDeepQ()

    info = lambda_info(request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id'],{})

    index_dict['info'] = add_file_to_info(info, files)

    if request.method == "GET":
        index_dict['form'] = form
        return render(request, "easyRL_app/index.html", context=index_dict)

    elif request.method == "POST":
        form = forms.HyperParameterFormDeepQ(request.POST)
        if form.is_valid():
            index_dict['form'] = form

        return render(request, "easyRL_app/index.html", context=index_dict)

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
            request.session['job_id'] = generate_jobID()
            # create ec2 instance
            debug_sessions(request)
            #lambda_create_instance(
            #    request.session['aws_access_key'],
            #    request.session['aws_secret_key'],
            #    request.session['aws_security_token'],
            #    request.session['job_id'],
            #    {}
            #)
            request.session['aws_succeed'] = True
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
        request.session['job_id'],
        {
            "instanceType": get_safe_value(str, request.POST.get("instanceType"), "c4.xlarge")
            ,"instanceID": get_safe_value(str, request.POST.get("instanceID"), "")
            ,"killTime": get_safe_value(int, request.POST.get("killTime"), 600)
            ,"environment": get_safe_value(int, request.POST.get("environment"), 1)
            ,"continuousTraining" : get_safe_value(str, request.POST.get("continuousTraining"), "False")
            ,"agent": get_safe_value(int, request.POST.get("agent"), 1)
            ,"episodes": get_safe_value(int, request.POST.get("episodes"), 20)
            ,"steps": get_safe_value(int, request.POST.get("steps"), 50)
            ,"gamma": get_safe_value(float, request.POST.get("gamma"), 0.97)
            ,"minEpsilon": get_safe_value(float, request.POST.get("minEpsilon"), 0.01)
            ,"maxEpsilon": get_safe_value(float, request.POST.get("maxEpsilon"), 0.99)
            ,"decayRate": get_safe_value(float, request.POST.get("decayRate"), 0.01)
            ,"batchSize": get_safe_value(int, request.POST.get("batchSize"), 32)
            ,"memorySize": get_safe_value(int, request.POST.get("memorySize"), 1000)
            ,"targetInterval": get_safe_value(int, request.POST.get("targetInterval"), 10)
            ,"alpha": get_safe_value(float, request.POST.get("alpha"), 0.9)
            ,"historyLength": get_safe_value(int, request.POST.get("historyLength"), 10)

            ,"delta": get_safe_value(int, request.POST.get("delta"), 0.001)
            ,"sigma": get_safe_value(int, request.POST.get("sigma"), 0.5)
            ,"population": get_safe_value(int, request.POST.get("population"), 10)
            ,"elite": get_safe_value(int, request.POST.get("elite"), 0.2)

            ,"tau": get_safe_value(int, request.POST.get("tau"), 0.97)
            ,"temperature": get_safe_value(int, request.POST.get("temperature"), 0.97)

            ,"learningRate": get_safe_value(int, request.POST.get("learningRate"), 0.001)
            ,"policyLearnRate": get_safe_value(int, request.POST.get("policyLearnRate"), 0.001)
            ,"valueLearnRate": get_safe_value(int, request.POST.get("valueLearnRate"), 0.001)
            ,"horizon": get_safe_value(int, request.POST.get("horizon"), 50)
            ,"epochSize": get_safe_value(int, request.POST.get("epochSize"), 500)
            ,"ppoEpsilon": get_safe_value(int, request.POST.get("ppoEpsilon"), 0.2)
            ,"ppoLambda": get_safe_value(int, request.POST.get("ppoLambda"), 0.95)
            ,"valueLearnRatePlus": get_safe_value(int, request.POST.get("valueLearnRatePlus"), 0.001)
        } 

    )
    # clear up all sessions
    for key in keys:
        del request.session[key]
    return HttpResponseRedirect("/easyRL_app/login/")

@csrf_exempt
def train(request):
    debug_sessions(request)
    if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
        return HttpResponse(apps.ERROR_UNAUTHENTICATED)
    print("{}request_parameters{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, debug_parameters(request)))
    return HttpResponse(lambda_run_job(
        request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id'],
        {
            "instanceType": get_safe_value(str, request.POST.get("c4.xlarge"), "c4.xlarge")
            ,"instanceID": get_safe_value(str, request.POST.get("instanceID"), "")
            ,"killTime": get_safe_value(int, request.POST.get("killTime"), 600)
            ,"environment": get_safe_value(int, request.POST.get("environment"), 1)
            ,"continuousTraining" : get_safe_value(str, request.POST.get("continuousTraining"), "False")
            ,"agent": get_safe_value(int, request.POST.get("agent"), 1)
            ,"episodes": get_safe_value(int, request.POST.get("episodes"), 20)
            ,"steps": get_safe_value(int, request.POST.get("steps"), 50)
            ,"gamma": get_safe_value(float, request.POST.get("gamma"), 0.97)
            ,"minEpsilon": get_safe_value(float, request.POST.get("minEpsilon"), 0.01)
            ,"maxEpsilon": get_safe_value(float, request.POST.get("maxEpsilon"), 0.99)
            ,"decayRate": get_safe_value(float, request.POST.get("decayRate"), 0.01)
            ,"batchSize": get_safe_value(int, request.POST.get("batchSize"), 32)
            ,"memorySize": get_safe_value(int, request.POST.get("memorySize"), 1000)
            ,"targetInterval": get_safe_value(int, request.POST.get("targetInterval"), 10)
            ,"alpha": get_safe_value(float, request.POST.get("alpha"), 0.9)
            ,"historyLength": get_safe_value(int, request.POST.get("historyLength"), 10)

            ,"delta": get_safe_value(int, request.POST.get("delta"), 0.001)
            ,"sigma": get_safe_value(int, request.POST.get("sigma"), 0.5)
            ,"population": get_safe_value(int, request.POST.get("population"), 10)
            ,"elite": get_safe_value(int, request.POST.get("elite"), 0.2)

            ,"tau": get_safe_value(int, request.POST.get("tau"), 0.97)
            ,"temperature": get_safe_value(int, request.POST.get("temperature"), 0.97)

            ,"learningRate": get_safe_value(int, request.POST.get("learningRate"), 0.001)
            ,"policyLearnRate": get_safe_value(int, request.POST.get("policyLearnRate"), 0.001)
            ,"valueLearnRate": get_safe_value(int, request.POST.get("valueLearnRate"), 0.001)
            ,"horizon": get_safe_value(int, request.POST.get("horizon"), 50)
            ,"epochSize": get_safe_value(int, request.POST.get("epochSize"), 500)
            ,"ppoEpsilon": get_safe_value(int, request.POST.get("ppoEpsilon"), 0.2)
            ,"ppoLambda": get_safe_value(int, request.POST.get("ppoLambda"), 0.95)
            ,"valueLearnRatePlus": get_safe_value(int, request.POST.get("valueLearnRatePlus"), 0.001)
        } 
    ))

@csrf_exempt
def test(request):
    debug_sessions(request)
    if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
        return HttpResponse(apps.ERROR_UNAUTHENTICATED)
    print("{}request_parameters{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, debug_parameters(request)))
    return HttpResponse(lambda_test_job(
        request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id'],
        {
            "instanceType": get_safe_value(str, request.POST.get("instanceType"), "c4.xlarge")
            ,"instanceID": get_safe_value(str, request.POST.get("instanceID"), "")
            ,"killTime": get_safe_value(int, request.POST.get("killTime"), 600)
            ,"environment": get_safe_value(int, request.POST.get("environment"), 1)
            ,"continuousTraining" : get_safe_value(str, request.POST.get("continuousTraining"), "False")
            ,"agent": get_safe_value(int, request.POST.get("agent"), 1)
            ,"episodes": get_safe_value(int, request.POST.get("episodes"), 20)
            ,"steps": get_safe_value(int, request.POST.get("steps"), 50)
            ,"gamma": get_safe_value(float, request.POST.get("gamma"), 0.97)
            ,"minEpsilon": get_safe_value(float, request.POST.get("minEpsilon"), 0.01)
            ,"maxEpsilon": get_safe_value(float, request.POST.get("maxEpsilon"), 0.99)
            ,"decayRate": get_safe_value(float, request.POST.get("decayRate"), 0.01)
            ,"batchSize": get_safe_value(int, request.POST.get("batchSize"), 32)
            ,"memorySize": get_safe_value(int, request.POST.get("memorySize"), 1000)
            ,"targetInterval": get_safe_value(int, request.POST.get("targetInterval"), 10)
            ,"alpha": get_safe_value(float, request.POST.get("alpha"), 0.9)
            ,"historyLength": get_safe_value(int, request.POST.get("historyLength"), 10)

            ,"delta": get_safe_value(int, request.POST.get("delta"), 0.001)
            ,"sigma": get_safe_value(int, request.POST.get("sigma"), 0.5)
            ,"population": get_safe_value(int, request.POST.get("population"), 10)
            ,"elite": get_safe_value(int, request.POST.get("elite"), 0.2)

            ,"tau": get_safe_value(int, request.POST.get("tau"), 0.97)
            ,"temperature": get_safe_value(int, request.POST.get("temperature"), 0.97)

            ,"learningRate": get_safe_value(int, request.POST.get("learningRate"), 0.001)
            ,"policyLearnRate": get_safe_value(int, request.POST.get("policyLearnRate"), 0.001)
            ,"valueLearnRate": get_safe_value(int, request.POST.get("valueLearnRate"), 0.001)
            ,"horizon": get_safe_value(int, request.POST.get("horizon"), 50)
            ,"epochSize": get_safe_value(int, request.POST.get("epochSize"), 500)
            ,"ppoEpsilon": get_safe_value(int, request.POST.get("ppoEpsilon"), 0.2)
            ,"ppoLambda": get_safe_value(int, request.POST.get("ppoLambda"), 0.95)
            ,"valueLearnRatePlus": get_safe_value(int, request.POST.get("valueLearnRatePlus"), 0.001)
        } 
    ))

@csrf_exempt
def poll(request):
    try:
        debug_sessions(request)
        if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
            return HttpResponse(apps.ERROR_UNAUTHENTICATED)
        print("{}request_parameters{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, debug_parameters(request)))
        response = HttpResponse(lambda_poll(
            request.session['aws_access_key'],
            request.session['aws_secret_key'],
            request.session['aws_security_token'],
            request.session['job_id'],
            {
                "instanceType": get_safe_value(str, request.POST.get("instanceType"), "c4.xlarge")
                ,"instanceID": get_safe_value(str, request.POST.get("instanceID"), "")
                ,"killTime": get_safe_value(int, request.POST.get("killTime"), 600)
                ,"continuousTraining" : get_safe_value(int, request.POST.get("continuousTraining"), 0)
                ,"environment": get_safe_value(int, request.POST.get("environment"), 1)
                ,"agent": get_safe_value(int, request.POST.get("agent"), 1)
                ,"episodes": get_safe_value(int, request.POST.get("episodes"), 20)
                ,"steps": get_safe_value(int, request.POST.get("steps"), 50)
                ,"gamma": get_safe_value(float, request.POST.get("gamma"), 0.97)
                ,"minEpsilon": get_safe_value(float, request.POST.get("minEpsilon"), 0.01)
                ,"maxEpsilon": get_safe_value(float, request.POST.get("maxEpsilon"), 0.99)
                ,"decayRate": get_safe_value(float, request.POST.get("decayRate"), 0.01)
                ,"batchSize": get_safe_value(int, request.POST.get("batchSize"), 32)
                ,"memorySize": get_safe_value(int, request.POST.get("memorySize"), 1000)
                ,"targetInterval": get_safe_value(int, request.POST.get("targetInterval"), 10)
                ,"alpha": get_safe_value(float, request.POST.get("alpha"), 0.9)
                ,"historyLength": get_safe_value(int, request.POST.get("historyLength"), 10)

                ,"delta": get_safe_value(int, request.POST.get("delta"), 0.001)
                ,"sigma": get_safe_value(int, request.POST.get("sigma"), 0.5)
                ,"population": get_safe_value(int, request.POST.get("population"), 10)
                ,"elite": get_safe_value(int, request.POST.get("elite"), 0.2)

                ,"tau": get_safe_value(int, request.POST.get("tau"), 0.97)
                ,"temperature": get_safe_value(int, request.POST.get("temperature"), 0.97)

                ,"learningRate": get_safe_value(int, request.POST.get("learningRate"), 0.001)
                ,"policyLearnRate": get_safe_value(int, request.POST.get("policyLearnRate"), 0.001)
                ,"valueLearnRate": get_safe_value(int, request.POST.get("valueLearnRate"), 0.001)
                ,"horizon": get_safe_value(int, request.POST.get("horizon"), 50)
                ,"epochSize": get_safe_value(int, request.POST.get("epochSize"), 500)
                ,"ppoEpsilon": get_safe_value(int, request.POST.get("ppoEpsilon"), 0.2)
                ,"ppoLambda": get_safe_value(int, request.POST.get("ppoLambda"), 0.95)
                ,"valueLearnRatePlus": get_safe_value(int, request.POST.get("valueLearnRatePlus"), 0.001)
            }                
        ))
        return response
    except:
        return {
            "instanceState": "booting",
            "instanceStateText": "Loading..."
        }

@csrf_exempt
def info(request):
    debug_sessions(request)
    if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
        return HttpResponse(apps.ERROR_UNAUTHENTICATED)
    print("{}request_parameters{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, debug_parameters(request)))
    return HttpResponse(lambda_info(
        request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id'],
        {}                
    ))

@csrf_exempt
def import_model_lambda(request):
    debug_sessions(request)
    if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
        return HttpResponse(apps.ERROR_UNAUTHENTICATED)
    print("{}request_parameters{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, debug_parameters(request)))
    return HttpResponseRedirect('/easyRL_app/',lambda_import(
        request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id'],
        {}                
    ))

@csrf_exempt
def export_model(request):
    debug_sessions(request)
    if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
        return HttpResponse(apps.ERROR_UNAUTHENTICATED)
    print("{}request_parameters{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, debug_parameters(request)))
    return HttpResponse(lambda_export_model(
        request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id'],
        {
            "instanceType": get_safe_value(str, request.POST.get("instanceType"), "c4.xlarge")
            ,"instanceID": get_safe_value(str, request.POST.get("instanceID"), "")
            ,"killTime": get_safe_value(int, request.POST.get("killTime"), 600)
            ,"environment": get_safe_value(int, request.POST.get("environment"), 1)
            ,"continuousTraining" : get_safe_value(str, request.POST.get("continuousTraining"), "False")
            ,"agent": get_safe_value(int, request.POST.get("agent"), 1)
            ,"episodes": get_safe_value(int, request.POST.get("episodes"), 20)
            ,"steps": get_safe_value(int, request.POST.get("steps"), 50)
            ,"gamma": get_safe_value(float, request.POST.get("gamma"), 0.97)
            ,"minEpsilon": get_safe_value(float, request.POST.get("minEpsilon"), 0.01)
            ,"maxEpsilon": get_safe_value(float, request.POST.get("maxEpsilon"), 0.99)
            ,"decayRate": get_safe_value(float, request.POST.get("decayRate"), 0.01)
            ,"batchSize": get_safe_value(int, request.POST.get("batchSize"), 32)
            ,"memorySize": get_safe_value(int, request.POST.get("memorySize"), 1000)
            ,"targetInterval": get_safe_value(int, request.POST.get("targetInterval"), 10)
            ,"alpha": get_safe_value(float, request.POST.get("alpha"), 0.9)
            ,"historyLength": get_safe_value(int, request.POST.get("historyLength"), 10)

            ,"delta": get_safe_value(int, request.POST.get("delta"), 0.001)
            ,"sigma": get_safe_value(int, request.POST.get("sigma"), 0.5)
            ,"population": get_safe_value(int, request.POST.get("population"), 10)
            ,"elite": get_safe_value(int, request.POST.get("elite"), 0.2)

            ,"tau": get_safe_value(int, request.POST.get("tau"), 0.97)
            ,"temperature": get_safe_value(int, request.POST.get("temperature"), 0.97)

            ,"learningRate": get_safe_value(int, request.POST.get("learningRate"), 0.001)
            ,"policyLearnRate": get_safe_value(int, request.POST.get("policyLearnRate"), 0.001)
            ,"valueLearnRate": get_safe_value(int, request.POST.get("valueLearnRate"), 0.001)
            ,"horizon": get_safe_value(int, request.POST.get("horizon"), 50)
            ,"epochSize": get_safe_value(int, request.POST.get("epochSize"), 500)
            ,"ppoEpsilon": get_safe_value(int, request.POST.get("ppoEpsilon"), 0.2)
            ,"ppoLambda": get_safe_value(int, request.POST.get("ppoLambda"), 0.95)
            ,"valueLearnRatePlus": get_safe_value(int, request.POST.get("valueLearnRatePlus"), 0.001)
        }
    ))

@csrf_exempt
def halt(request):
    debug_sessions(request)
    if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
        return HttpResponse(apps.ERROR_UNAUTHENTICATED)
    print("{}request_parameters{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, debug_parameters(request)))
    return HttpResponse(lambda_halt_job(
        request.session['aws_access_key'],
        request.session['aws_secret_key'],
        request.session['aws_security_token'],
        request.session['job_id'],
        {
            "instanceType": get_safe_value(str, request.POST.get("instanceType"), "c4.xlarge")
            ,"instanceID": get_safe_value(str, request.POST.get("instanceID"), "")
            ,"killTime": get_safe_value(int, request.POST.get("killTime"), 600)
            ,"environment": get_safe_value(int, request.POST.get("environment"), 1)
            ,"continuousTraining" : get_safe_value(str, request.POST.get("continuousTraining"), "False")
            ,"agent": get_safe_value(int, request.POST.get("agent"), 1)
            ,"episodes": get_safe_value(int, request.POST.get("episodes"), 20)
            ,"steps": get_safe_value(int, request.POST.get("steps"), 50)
            ,"gamma": get_safe_value(float, request.POST.get("gamma"), 0.97)
            ,"minEpsilon": get_safe_value(float, request.POST.get("minEpsilon"), 0.01)
            ,"maxEpsilon": get_safe_value(float, request.POST.get("maxEpsilon"), 0.99)
            ,"decayRate": get_safe_value(float, request.POST.get("decayRate"), 0.01)
            ,"batchSize": get_safe_value(int, request.POST.get("batchSize"), 32)
            ,"memorySize": get_safe_value(int, request.POST.get("memorySize"), 1000)
            ,"targetInterval": get_safe_value(int, request.POST.get("targetInterval"), 10)
            ,"alpha": get_safe_value(float, request.POST.get("alpha"), 0.9)
            ,"historyLength": get_safe_value(int, request.POST.get("historyLength"), 10)

            ,"delta": get_safe_value(int, request.POST.get("delta"), 0.001)
            ,"sigma": get_safe_value(int, request.POST.get("sigma"), 0.5)
            ,"population": get_safe_value(int, request.POST.get("population"), 10)
            ,"elite": get_safe_value(int, request.POST.get("elite"), 0.2)

            ,"tau": get_safe_value(int, request.POST.get("tau"), 0.97)
            ,"temperature": get_safe_value(int, request.POST.get("temperature"), 0.97)

            ,"learningRate": get_safe_value(int, request.POST.get("learningRate"), 0.001)
            ,"policyLearnRate": get_safe_value(int, request.POST.get("policyLearnRate"), 0.001)
            ,"valueLearnRate": get_safe_value(int, request.POST.get("valueLearnRate"), 0.001)
            ,"horizon": get_safe_value(int, request.POST.get("horizon"), 50)
            ,"epochSize": get_safe_value(int, request.POST.get("epochSize"), 500)
            ,"ppoEpsilon": get_safe_value(int, request.POST.get("ppoEpsilon"), 0.2)
            ,"ppoLambda": get_safe_value(int, request.POST.get("ppoLambda"), 0.95)
            ,"valueLearnRatePlus": get_safe_value(int, request.POST.get("valueLearnRatePlus"), 0.001)
        }
    ))

'''
def lambda_create_instance(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_CREATE_INSTANCE,
        "arguments": arguments,
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    print("{}lambda_create_instance{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, response['Payload'].read()))
    if response['StatusCode'] == 200:
        streambody = response['Payload'].read().decode()
        print("{}stream_body{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, streambody))
        return True
    return False
'''
def lambda_import(aws_access_key, aws_secret_key, aws_security_token, job_id,arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_IMPORT,
        "arguments": arguments,
    }
    
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    print("{}lambda_terminate_instance{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, response['Payload'].read()))
    if response['StatusCode'] == 200:
        streambody = response['Payload'].read().decode()
        print("{}stream_body{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, streambody))
        return True
    return False

def lambda_terminate_instance(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_TERMINAL_INSTANCE,
        "arguments": arguments,
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    print("{}lambda_terminate_instance{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, response['Payload'].read()))
    if response['StatusCode'] == 200:
        streambody = response['Payload'].read().decode()
        print("{}stream_body{}={}".format(apps.FORMAT_BLUE, apps.FORMAT_RESET, streambody))
        return True
    return False

def lambda_halt_job(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_HALT_JOB,
        "arguments": arguments
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    payload = response['Payload'].read()
    print("{}lambda_halt_job{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, payload))
    if len(payload) != 0:
        return "{}".format(payload)[2:-1]
    else:
        return ""

def lambda_export_model(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_EXPORT_MODEL,
        "arguments": arguments
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    payload = response['Payload'].read()
    print("{}lambda_export_model{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, payload))
    if len(payload) != 0:
        return "{}".format(payload)[2:-1]
    else:
        return ""

def lambda_poll(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_POLL,
        "arguments": arguments,
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    payload = response['Payload'].read()
    print("{}lambda_poll{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, payload))
    if len(payload) != 0:
        return "{}".format(payload)[2:-1]
    else:
        return ""

def lambda_run_job(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_RUN_JOB,
        "arguments": arguments,
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    payload = response['Payload'].read()
    print("{}lambda_run_job{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, payload))
    if len(payload) != 0:
        return "{}".format(payload)[2:-1]
    else:
        return ""

def lambda_test_job(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_RUN_TEST,
        "arguments": arguments,
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    payload = response['Payload'].read()
    print("{}lambda_test_job{}={}".format(apps.FORMAT_RED, apps.FORMAT_RESET, payload))
    if len(payload) != 0:
        return "{}".format(payload)[2:-1]
    else:
        return ""

def get_safe_value_bool(boolean_val):
    if boolean_val == 'True':
        return True
    else:
        return False

def get_safe_value(convert_function, input_value, default_value):
    try:
        return convert_function(input_value)
    except ValueError as _:
        return default_value
    except Exception as _:
        return default_value

def debug_parameters(request):
    return ' '.join(["{}={}".format(key, value) for key, value in request.POST.items()])

def debug_sessions(request):
    for key in request.session.keys():
        print("{}{}{}={}".format(apps.FORMAT_CYAN, key, apps.FORMAT_RESET, request.session[key]))

def lambda_info(aws_access_key, aws_secret_key, aws_security_token, job_id, arguments):
    lambdas = get_aws_lambda(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    data = {
        "accessKey": aws_access_key,
        "secretKey": aws_secret_key,
        "sessionToken": aws_security_token,
        "jobID": job_id,
        "task": apps.TASK_INFO,
        "arguments": arguments,
    }
    response = invoke_aws_lambda_func(lambdas, str(data).replace('\'','"'))
    payload = response['Payload'].read()
    print("{}lambda_info_job{}={}".format(apps.FORMAT_GREEN, apps.FORMAT_RESET, payload))
    if len(payload) != 0:
        return "{}".format(payload)[2:-1]
    else:
        return ""

def add_file_to_info(payload, files):
    result = json.loads(payload)
    for val in result['environments']:
        for file in files:
            if val['name'] == 'Cart Pole':
                val['file'] = 'Cart Pole.jpg'
                continue
        
            if val['name'].replace('.','').replace(' ', '').lower() in file.replace('_','').replace(' ','').lower():
                val['file'] = file
                break

    return result
    
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from .models import Document
class import_model(CreateView):
    model = Document
    fields = ['upload', ]
    success_url = reverse_lazy('upload')

from django.views import View
from storages.backends.s3boto3 import S3Boto3Storage
class file_upload(View):
    def post(self, request, **kwargs):
        debug_sessions(request)
        if 'aws_succeed' not in request.session or not request.session['aws_succeed']:
            return HttpResponseRedirect("/easyRL_app/login/")
        file_obj = request.FILES.get('upload', 'EMPTY')
        aws_access_key = request.session['aws_access_key']
        aws_secret_key = request.session['aws_secret_key']
        aws_security_token = request.session['aws_security_token']
        job_id = request.session['job_id']
        bucket = "easyrl-{}{}".format(job_id, request.POST.get('session', '0'))

        media_storage = S3Boto3Storage()
        media_storage.location = ''
        media_storage.file_overwrite = True
        media_storage.access_key = aws_access_key
        media_storage.secret_key = aws_secret_key
        media_storage.bucket_name = bucket


        # os.rename(os.path.join(media_storage.location,file_obj.name),os.path.join(media_storage.location,'model.bin'))
        # file_obj.name = 'model.bin'
        s3_file_path = os.path.join(
            media_storage.location,
            'model.bin'
        )
        
        media_storage.save(s3_file_path, file_obj)

        
        #file_url = media_storage.url(s3_file_path) # direct path of uploaded file on s3
        return HttpResponse(lambda_import(aws_access_key, aws_secret_key, aws_security_token, job_id, {}), status=200)
  
    
