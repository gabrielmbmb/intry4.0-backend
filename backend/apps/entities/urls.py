from django.urls import path, re_path
from backend.apps.entities import views


urlpatterns = [
    re_path(r"^entities/?$", views.EntitiesView.as_view(), name="entities",),
    path(r"entities/<str:urn>/", views.EntitiesView.as_view(), name="entity",),
]
