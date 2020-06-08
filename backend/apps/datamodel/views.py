from rest_framework import viewsets
from backend.apps.datamodel.models import DataModel
from backend.apps.datamodel.serializers import DataModelSerializer


class DataModelViewSet(viewsets.ModelViewSet):
    queryset = DataModel.objects.all()
    serializer_class = DataModelSerializer
