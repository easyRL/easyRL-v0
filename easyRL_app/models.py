from django.db import models
from core.storage import MediaStorage

# Create your models here.
class Document(models.Model):
    upload = models.FileField(verbose_name="")
#     upload = models.FileField(verbose_name="", storage=MediaStorage())
    #uploaded_at = models.DateTimeField(auto_now_add=True)
#     def __init__(self, location, aws_access_key, aws_secret_key, bucket_name):
#         self.uploaded_at = models.DateTimeField(auto_now_add=True)
#         self.upload = models.FileField(
#             verbose_name="",
#             storage=MediaStorage(
#                 location,
#                 aws_access_key,
#                 aws_secret_key,
#                 bucket_name
#         ))
# class Document(models.Model):

#     def __init__(self, location, aws_access_key, aws_secret_key, bucket_name):
#         self.upload = models.FileField(
#             verbose_name="",
#             storage=MediaStorage(
#                 location,
#                 aws_access_key,
#                 aws_secret_key,
#                 bucket_name
#             )
#         )

    @classmethod
    def save_get(cls, *args, **kwargs):
#         location = kwargs.items()["location"]
#         aws_access_key = kwargs.items()["aws_access_key"]
#         aws_secret_key = kwargs.items()["aws_secret_key"]
#         bucket_name = kwargs.items()["bucket_name"]

        location = args[0]
        aws_access_key = args[1]
        aws_secret_key = args[2]
        bucket_name = args[3]
        
        print("GET", aws_access_key)

        Document.upload = models.FileField(
            verbose_name="", 
            storage=MediaStorage(location, aws_access_key, aws_secret_key, bucket_name))
        #Document(cls).save(*args, **kwargs)
    
    @classmethod
    def save_post(cls, *args, **kwargs):
#         location = kwargs.items()["location"]
#         aws_access_key = kwargs.items()["aws_access_key"]
#         aws_secret_key = kwargs.items()["aws_secret_key"]
#         bucket_name = kwargs.items()["bucket_name"]

        location = args[0]
        aws_access_key = args[1]
        aws_secret_key = args[2]
        bucket_name = args[3]
        
        print("POST", aws_access_key)

        Document.upload = models.FileField(
            verbose_name="", 
            storage=MediaStorage(location, aws_access_key, aws_secret_key, bucket_name))
    
        