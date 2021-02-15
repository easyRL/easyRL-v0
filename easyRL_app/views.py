from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def index(request):
    my_dict = {"insert_me":"This is the insert from easyRL.views"}
    return render(request, "easyRL_app/index.html", context=my_dict)
