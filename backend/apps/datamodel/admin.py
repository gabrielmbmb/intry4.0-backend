from django.contrib import admin
from backend.apps.datamodel import models

admin.site.register(models.DataModel)
admin.site.register(models.TrainFile)
