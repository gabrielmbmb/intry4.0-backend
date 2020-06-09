from django.contrib import admin
from backend.apps.datamodel import models

admin.site.register(models.DataModel)
admin.site.register(models.PLCModel)
admin.site.register(models.SensorModel)
admin.site.register(models.PCAMahalanobisModel)
admin.site.register(models.AutoencoderModel)
admin.site.register(models.KMeansModel)
admin.site.register(models.OneClassSVMModel)
admin.site.register(models.GaussianDistributionModel)
admin.site.register(models.IsolationForestModel)
admin.site.register(models.LocalOutlierFactorModel)
admin.site.register(models.KNNModel)
