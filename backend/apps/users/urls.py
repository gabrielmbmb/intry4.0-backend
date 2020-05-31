from django.urls import re_path, include
from rest_framework.routers import DefaultRouter
from backend.apps.users import views

router = DefaultRouter()
router.register(r"", views.UserViewSet)

urlpatterns = [
    re_path(r"^users/", include(router.urls)),
]
