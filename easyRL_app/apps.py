from django.apps import AppConfig

TASK_CREATE_INSTANCE = "createInstance"
TASK_EXPORT_MODEL = "exportModel"
TASK_TERMINAL_INSTANCE = "terminateInstance"
TASK_RUN_JOB = "runJob"
TASK_RUN_TEST = "runTest"
TASK_POLL = "poll"
TASK_IS_JOB_RUNNING = "isRunning"
TASK_HALT_JOB = "haltJob"
TASK_INFO = "info"
TASK_IMPORT = "import"

ERROR_NONE = 0
ERROR_UNAUTHENTICATED = 1

# https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
FORMAT_HEADER = '\033[95m'
FORMAT_RED = "\033[1;31m"  
FORMAT_BLUE = "\033[1;34m"
FORMAT_CYAN = "\033[1;36m"
FORMAT_GREEN = "\033[0;32m"
FORMAT_WARNING = '\033[93m'
FORMAT_FAIL = '\033[91m'
FORMAT_RESET = "\033[0;0m"
FORMAT_BOLD = "\033[;1m"
FORMAT_UNDERLINE = '\033[4m'

IMAGE_FILE = "Training-episode-{}.gif"
DATA_JSON_FILE = "data.json"
LOCAL_JSON_FILE = "/tmp/{}.json"


class EasyrlAppConfig(AppConfig):
    name = 'easyRL_app'
