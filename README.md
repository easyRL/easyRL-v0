# How to Run EasyRL Cloud

This application uses Django as a webserver to host the interface. To run the server you must have Python with the following packages installed:

- django
- boto3
- botocore

`python3.7 -m pip install django boto3 botocore`

Once setup, start the server by running manage.py This will default to port 8000:

`python3.7 manage.py runserver [optional port number]`

Once the server is running, open the webpage below in your browser and log in with AWS credentials to begin!

`http://127.0.0.1:8000/easyRL_app/`
