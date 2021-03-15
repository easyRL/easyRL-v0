from django.db import models

# Create your models here.
class Document(models.Model):
    upload = models.FileField(verbose_name="")