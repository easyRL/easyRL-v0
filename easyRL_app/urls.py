from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from . import  views as v
from django.contrib.auth import views

urlpatterns = [
    path('', v.index, name="index"),
    path('login/', v.login, name='login'),
    path('logout/', v.logout, name='logout'),
    path('test_ci/', v.test_create_instance, name='create_instance'),
    path('test_ti/', v.test_terminate_instance, name='terminate_instance'),
    path('test_rj/', v.test_run_job, name='run_job'),
]
