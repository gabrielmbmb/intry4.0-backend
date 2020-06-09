from drf_yasg.utils import swagger_auto_schema
from django.utils.decorators import method_decorator
from rest_framework import viewsets, status, permissions
from rest_framework.response import Response
from rest_framework.decorators import action
from backend.apps.datamodel.models import DataModel
from backend.apps.datamodel.serializers import DataModelSerializer


@method_decorator(
    name="list",
    decorator=swagger_auto_schema(
        operation_description="Returns the list of created Anomaly Detection models."
    ),
)
@method_decorator(
    name="retrieve",
    decorator=swagger_auto_schema(
        operation_description="Returns an Anomaly Detection model."
    ),
)
@method_decorator(
    name="create",
    decorator=swagger_auto_schema(
        operation_description="Creates an Anomaly Detection model."
    ),
)
@method_decorator(
    name="update",
    decorator=swagger_auto_schema(
        operation_description="Updates an Anomaly Detection model."
    ),
)
@method_decorator(
    name="partial_update",
    decorator=swagger_auto_schema(
        operation_description="Makes a partial update of an Anomaly Detection model."
    ),
)
@method_decorator(
    name="destroy",
    decorator=swagger_auto_schema(
        operation_description="Removes an Anomaly Detection model."
    ),
)
class DataModelViewSet(viewsets.ModelViewSet):
    queryset = DataModel.objects.all()
    serializer_class = DataModelSerializer
    permission_classes = (permissions.IsAuthenticated,)

    @action(detail=True, methods=["POST"])
    def train(self, request, *args, **kwargs):
        """Triggers the training of an Anomaly Detection Model."""
        datamodel = self.get_object()
        return Response(data={"detail": "ah shit"}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["POST"])
    def deploy(self, request, *args, **kwargs):
        """Deploys an Anomaly Detection Model."""
        pass

    @action(detail=True, methods=["POST"])
    def predict(self, request, *args, **kwargs):
        """Receives new measures from """
        pass
