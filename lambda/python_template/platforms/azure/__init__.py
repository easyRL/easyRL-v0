import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import json
import logging
import azure.functions as func
from . import handler

#
# Azure Functions Default Function
#
# This hander is used as a bridge to call the platform neutral
# version in handler.py. This script is put into the scr directory
# when using publish.sh.
#
# @param request
#
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    result = handler.yourFunction(req.get_json(), None)
    return func.HttpResponse(str(result).replace("'", '"'), status_code=200)

#    name = req.params.get('name')
#    if not name:
#        try:
#            req_body = req.get_json()
#        except ValueError:
#            pass
#        else:
#            name = req_body.get('name')

#    if name:
#        return func.HttpResponse(f"Hello {name}!")
#    else:
#        return func.HttpResponse(
#             "Please pass a name on the query string or in the request body",
#             status_code=400
#        )
