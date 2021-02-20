from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from . import  views as view

urlpatterns = [

    path('', view.index, name="index"),
    path('login/', view.login, name='login'),
    path('logout/', view.logout, name='logout'),
    path('test/', view.test_data, name='test'),
    path('test_ci/', view.test_create_instance, name='create_instance'),
    path('test_ti/', view.test_terminate_instance, name='terminate_instance'),
]
