from django.urls import re_path, include
from rest_framework.routers import DefaultRouter
from backend.apps.datamodel import views

router = DefaultRouter()
router.register(r"", views.DataModelViewSet)

urlpatterns = [
    re_path(r"^datamodels/", include(router.urls)),
]
