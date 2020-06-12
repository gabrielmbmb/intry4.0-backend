"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path, include
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
    openapi.Info(
        title="InTry 4.0 - Backend",
        default_version="v1",
        description="The documentation of the main backend of the project InTry 4.0",
        contact=openapi.Contact(email="gmartin_b@usal.es"),
        license=openapi.License(name="GPLv3"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path("admin/", admin.site.urls),
    # login page,
    re_path(r"^accounts/", include("django.contrib.auth.urls")),
    # users
    re_path(r"^api/v1/", include("backend.apps.users.urls")),
    # entities
    re_path(r"^api/v1/", include("backend.apps.entities.urls")),
    # datamodel
    re_path(r"^api/v1/", include("backend.apps.datamodel.urls")),
    # Custom user info Oauth2
    re_path(r"^o/", include("backend.apps.core.urls")),
    # Oauth2
    re_path(r"^o/", include("oauth2_provider.urls", namespace="oauth2_provider")),
    # swagger
    re_path(
        r"^swagger(?P<format>\.json|\.yaml)$",
        schema_view.without_ui(cache_timeout=0),
        name="schema-json",
    ),
    re_path(
        r"^swagger/$",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui",
    ),
    re_path(
        r"^redoc/$", schema_view.with_ui("redoc", cache_timeout=0), name="schema-redoc"
    ),
]
