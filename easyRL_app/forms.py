from django import forms
import uuid

class LoginForm (forms.Form):
    access_key_id = forms.TextInput(label="AWS ACCESS KEY ID", max_length=100)
    secret_key =  forms.TextInput(label='AWS SECRET ACCESS KEY', max_length=100)
    session_token = uuid.uuid4()