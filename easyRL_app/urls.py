from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from . import  views as view

urlpatterns = [
    path('', view.index, name="index"),
    path('login/', view.login, name='login'),
    path('logout/', view.logout, name='logout'),
    path('train/', view.train, name='train'),
    path('test/', view.test_data, name='test'),
]
