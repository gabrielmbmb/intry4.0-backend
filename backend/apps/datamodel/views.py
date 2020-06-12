from drf_yasg.utils import swagger_auto_schema
from django.utils.decorators import method_decorator
from rest_framework import viewsets, status, permissions
from rest_framework.exceptions import ParseError
from rest_framework.response import Response
from rest_framework.decorators import action
from backend.apps.datamodel.models import DataModel, TrainFile
from backend.apps.datamodel.serializers import (
    DataModelSerializer,
    DataModelTrainSerializer,
    TrainFileSerializer,
)


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

    def create(self, request):
        """Creates a datamodel."""
        serializer = DataModelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data

        model = DataModel(**validated_data)
        model.save()
        model.create_blackbox()

        return Response(data=validated_data, status=status.HTTP_200_OK)

    @action(
        detail=True, methods=["POST"], serializer_class=DataModelTrainSerializer,
    )
    def train(self, request, pk, *args, **kwargs):
        """Triggers the training of an Anomaly Detection Model with data from CrateDB"""
        serializer = DataModelTrainSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            datamodel = DataModel.objects.get(pk=pk)
        except DataModel.DoesNotExist:
            return Response(
                data={"detail": "There is no datamodel with the specified id"},
                status=status.HTTP_404_NOT_FOUND,
            )

        data = serializer.validated_data
        datamodel.train(with_source="db", **data)

        return Response(
            data={
                "detail": "Training has been initiated",
                "task_status": datamodel.task_status,
            },
            status=status.HTTP_202_ACCEPTED,
        )

    @action(
        detail=True,
        methods=["PUT"],
        url_path="train/csv",
        serializer_class=TrainFileSerializer,
    )
    def train_with_csv(self, request, pk, *args, **kwargs):
        """
        Triggers the training of an Anomaly Detection Model with data from the
        provided CSV.
        """
        try:
            datamodel = DataModel.objects.get(pk=pk)
        except DataModel.DoesNotExist:
            datamodel = None

        if datamodel:
            # validate data
            serializer = TrainFileSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            data = serializer.validated_data

            # create model
            model = TrainFile()
            model.file = data.get("file")
            model.datamodel = datamodel
            model.save()

            return Response(
                data={"detail": f"The file {model.file} has been uploaded"},
                status=status.HTTP_201_CREATED,
            )
        else:
            return Response(
                data={"detail": "There is no datamodel with the specified id"},
                status=status.HTTP_404_NOT_FOUND,
            )

    @action(
        detail=True, methods=["POST"], url_path="train/finished", serializer_class=None,
    )
    def train_finished(self, request, *args, **kwargs):
        """Endpoint to get notified that the trainning of a model has ended."""
        datamodel = self.get_object()
        datamodel.set_trained()
        return Response(status=status.HTTP_200_OK)

    @action(
        detail=True, methods=["POST"], url_path="train/deploy", serializer_class=None,
    )
    def deploy(self, request, *args, **kwargs):
        """Deploys an Anomaly Detection Model."""
        datamodel = self.get_object()
        datamodel.set_deployed()
        return Response(status=status.HTTP_200_OK)

    @action(detail=True, methods=["POST"], serializer_class=None)
    def predict(self, request, *args, **kwargs):
        """Receives new measures from """
        pass
