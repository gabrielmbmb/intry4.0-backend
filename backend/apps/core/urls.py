from django.urls import re_path
from backend.apps.core import views


urlpatterns = [
    re_path(r"^user_info/?$", views.UserInfo.as_view(), name="users-info",),
]
