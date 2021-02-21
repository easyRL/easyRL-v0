from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from . import  views as v
from django.contrib.auth import views

urlpatterns = [
    path('', v.index, name="index"),
    path('login/', v.login, name='login'),
    path('logout/', v.logout, name='logout'),
    path('train/', v.train, name='train'),
    path('test/', v.test_data, name='test'),
]
