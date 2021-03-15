from django import forms
from django.db import models

class AwsCredentialForm(forms.Form):
    aws_access_key = forms.CharField(label="AWS Access Key", widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    aws_secret_key = forms.CharField(label="AWS Secret Key",widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    aws_security_token = forms.CharField(label="AWS Security Token",widget=forms.PasswordInput(attrs={'class': 'form-control'}), required=False)

class HyperParameterBase(forms.Form):
    upload = models.FileField(verbose_name="")
    max_size  = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"max-size","name":"max-size","value":"200" }))    
    num_episodes = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"num-episodes","name":"num-episodes","value":"1000" }))
    gamma = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, "step":0.001, 'class': 'col-md-4 form-range slider', 'id':'gamma-slider', 'name':'gamma-slider','value': '0.970'}))
    gamma_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'gamma-text','class':  'form-control','readonly':'','value': '0.970'}))
    min_epsilon = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, "step": 0.001, 'class': "form-range slider", 'id':'min-slider', 'name':'min-slider','value': '.001'}))
    min_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'min-text','class':'form-control','readonly':'','value': '.001'}))
    max_epsilon = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, "step": 0.001, 'class': "form-range slider", 'id':'max-slider', 'name':'max-slider','value': '1'}))
    max_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'max-text','class':'form-control','readonly':'','value': '1'}))
    decay_rate = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 0.2, 'step': 0.001,'class': "form-range slider", 'id':'decay-slider', 'name':'decay-slider','value': '0.018'}))
    decay_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'decay-text','class':'form-control','readonly':'','value': '0.018'}))
    display_ep_speed = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 20, "step": 0.001, 'class': "form-range slider", 'id':'display-slider', 'name':'display-slider','value': '10'}))
    display_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'display-text','class':'form-control','readonly':'','value': '10'}))

class HyperParametersActorCritcForm(HyperParameterBase):
    horizon = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"horizon","name":"horizon","value":"1000" }))
    epoch = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"epoch","name":"epoch","value":"1000" })) 
    policy_learn_rate = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, 'step':0.001,'class': "form-range slider", 'id':'plr-slider', 'name':'plf-slider','value': '.018'}))
    plr_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'batch-plr','class':'form-control','readonly':'','value': '.018'}))
    value_learn_rate = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, 'step':0.001, 'class': "form-range slider", 'id':'vlr-slider', 'name':'vlr-slider','value': '.018'}))
    vlr_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'vlr-text','class':'form-control','readonly':'','value': '0.18'}))

class HyperParameterReinforceNativeForm(HyperParameterBase):
    policy_learn_rate = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 1, 'step':0.001,'class': "form-range slider", 'id':'plr-slider', 'name':'plf-slider','value': '.018'}))
    plr_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'batch-plr','class':'form-control','readonly':'','value': '.018'}))

class HyperParameterSARSA(HyperParameterBase):
    alpha = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 0.2, 'step': 0.001,'class': "form-range slider", 'id':'alpha-slider', 'name':'alpha-slider','value': '0.018'}))
    alpha_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'alpha-text','class':'form-control','readonly':'','value': '0.018'}))

class HyperParameterFormDeepQ(HyperParameterBase):
    target_update =forms.CharField(label="Update",widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"target-update","name":"target-update","value":"200" }))
    max_memory = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"max-memory","name":"max-memory","value":"1000" }))
    num_episodes = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"num-episodes","name":"num-episodes","value":"1000" }))
    batch = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 256, 'class': "form-range slider", 'id':'batch-slider', 'name':'batch-slider','value': '32'}))
    batch_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'batch-text','class':'form-control','readonly':'','value': '32'}))

class HyperParameterPPONativeForm(HyperParametersActorCritcForm):
    batch = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 256, 'class': "form-range slider", 'id':'batch-slider', 'name':'batch-slider','value': '32'}))
    batch_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'batch-text','class':'form-control','readonly':'','value': '32'}))    

class HyperParametersDoubleDuelingForm(HyperParameterFormDeepQ):
    learning_rate = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 100,  'step':0.001, 'class': "form-range slider", 'id':'learning-slider', 'name':'learing-slider','value': '32'}))
    learn_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'learning-text','class':'form-control','readonly':'','value': '0.001'}))   

class HyperParameterDRQNForm(HyperParameterFormDeepQ):
    history_length = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"history-length","name":"history-length","value":"10" }))

class HyperParameterQLearningForm(HyperParameterBase):
    alpha = forms.CharField(widget=forms.TextInput(attrs={'type': 'range', 'min': 0, 'max': 0.2, 'step': 0.001,'class': "form-range slider", 'id':'alpha-slider', 'name':'alpha-slider','value': '0.018'}))
    alpha_text = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "id":'alpha-text','class':'form-control','readonly':'','value': '0.018'}))

class HyperParameterConvDRQNForm(HyperParametersDoubleDuelingForm):
    history_length = forms.CharField(widget=forms.TextInput(attrs={"type":"text", "class":"form-control form-text-box","id":"history-length","name":"history-length","value":"10" }))
