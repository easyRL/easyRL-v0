# How to Run EasyRL Cloud

This application uses Django as a webserver to host the interface. To run the server you must have Python with the following packages installed:

- django
- django storage
- boto3
- botocore

`python3.7 -m pip install django boto3 botocore django-storages`

Before starting the server, we need to set up the environment variables for the WebGUI (The key must be the same as the one using AWS Lambda)

`export AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=xXxXXxxXxXxxXXXxXxxXXxxXXXxxxXxXxxxxxxX`

Make migration and migrate project relative to the model modification (required only when the code in models.py changed)

`python3 manage.py makemigrations
python3 manage.py migrate`

Once setup, start the server by running manage.py This will default to port 8000:

`python3.7 manage.py runserver [optional port number]`

Once the server is running, open the webpage below in your browser and log in with AWS credentials to begin!

`http://SERVER_IP:SERVER_PORT/easyRL_app/`

![Login](login.png "Login")
