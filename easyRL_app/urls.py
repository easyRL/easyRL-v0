from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from . import  views as view

urlpatterns = [
    path('', view.index, name="index")
    ,path('login/', view.login, name='login')
    ,path('logout/', view.logout, name='logout')
    ,path('train/', view.train, name='train')
    ,path('halt/', view.halt, name='halt')
    ,path('poll/', view.poll, name='poll')
    ,path('info/', view.info, name='info')
    ,path('export/', view.export_model, name="export_model")
    ,path('import/',view.import_model, name="import_model")
]
