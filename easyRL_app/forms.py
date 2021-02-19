from django import forms

import uuid

class AwsCredentialForm(forms.Form):
    aws_secret_key = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    aws_access_key = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    aws_security_token = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))

class HyperParameterForm(forms.Form):

    num_episodes = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"num-episodes","name":"num-episodes" }))
    max_memory = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"max-memory","name":"max-memory" }))
    max_size  = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"max-size","name":"max-size" }))
    target_update =forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"target-update","name":"target-update" }))
    
    gamma = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, "step":0.001, 'class': 'col-md-4 form-range slider', 'id':'gamma-slider', 'name':'gamma-slider'}))
    gamma_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'gamma-text','class':  'col-md-4 form-control'}))

    batch = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 2000, 'class': "form-range slider", 'id':'batch-slider', 'name':'batch-slider'}))
    batch_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'batch-text','class':'form-control'}))

    min_epsilon = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, "step": 0.001, 'class': "form-range slider", 'id':'min-slider', 'name':'min-slider'}))
    min_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'min-text','class':'form-control'}))

    max_epsilon = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, "step": 0.001, 'class': "form-range slider", 'id':'max-slider', 'name':'max-slider'}))
    max_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'max-text','class':'form-control'}))

    decay_rate = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 2000, 'class': "form-range slider", 'id':'decay-slider', 'name':'decay-slider'}))
    decay_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'decay-text','class':'form-control'}))

    display_ep_speed = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 2000, 'class': "form-range slider", 'id':'display-slider', 'name':'display-slider'}))
    display_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'display-text','class':'form-control'}))
