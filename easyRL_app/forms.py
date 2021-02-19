from django import forms
from django.contrib.auth.forms import AuthenticationForm
import uuid

class FormName(forms.Form):
    aws_secret_key = forms.CharField(widget=forms.PasswordInput())
    aws_access_key = forms.CharField(widget=forms.PasswordInput())
    aws_security_token = forms.CharField(widget=forms.PasswordInput())
    