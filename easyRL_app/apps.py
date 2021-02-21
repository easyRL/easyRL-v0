from django.apps import AppConfig

TASK_CREATE_INSTANCE = "createInstance"
TASK_TERMINAL_INSTANCE = "createInstance"
TASK_RUN_JOB = "runJob"

ERROR_NONE = 0
ERROR_UNAUTHENTICATED = 1

class EasyrlAppConfig(AppConfig):
    name = 'easyRL_app'
