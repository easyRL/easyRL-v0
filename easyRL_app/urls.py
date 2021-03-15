from django.urls import path
from . import  views as view
from easyRL_app import views

urlpatterns = [
    path('', view.index, name="index")
    ,path('login/', view.login, name='login')
    ,path('logout/', view.logout, name='logout')
    ,path('train/', view.train, name='train')
    ,path('test/', view.test, name='test')
    ,path('halt/', view.halt, name='halt')
    ,path('poll/', view.poll, name='poll')
    ,path('info/', view.info, name='info')
    ,path('export/', view.export_model, name="export_model")
    ,path('import/',views.import_model.as_view(), name="import_model")
    ,path('upload/',views.file_upload.as_view(), name="upload")
]
