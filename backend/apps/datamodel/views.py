from drf_yasg.utils import swagger_auto_schema
from django.utils.decorators import method_decorator
from rest_framework import viewsets, status, permissions
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework_extensions.mixins import NestedViewSetMixin
from backend.apps.datamodel.models import DataModel, TrainFile, DataModelPrediction
from backend.apps.datamodel.serializers import (
    DataModelSerializer,
    DataModelTrainSerializer,
    TrainFileSerializer,
    DataModelPredictionSerializer,
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
class DataModelViewSet(NestedViewSetMixin, viewsets.ModelViewSet):
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
        training_initiated = datamodel.train(with_source="db", **data)

        if not training_initiated:
            return Response(
                data={
                    "detail": f"The datamodel with id {datamodel.id} is already being "
                    "trained or the query did not return any row."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

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

            # check that the CSV file has the necessary columns to train the model
            file = data.get("file")
            index_column = data.get("index_column", None)

            (csv_was_valid, df) = datamodel.check_csv_columns(file, index_column)
            if csv_was_valid:
                # create model
                model = TrainFile()
                model.file = data.get("file")
                model.datamodel = datamodel
                model.save()

                training_initiated = datamodel.train(with_source="csv", train_df=df)

                if not training_initiated:
                    return Response(
                        data={
                            "detail": f"The datamodel with id {datamodel.id} is already being trained"
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                return Response(
                    data={
                        "detail": f"The file {model.file} has been uploaded and the "
                        f"training process for datamodel with id {datamodel.id} has "
                        "been initiated.",
                        "task_status": datamodel.task_status,
                    },
                    status=status.HTTP_201_CREATED,
                )
            else:
                return Response(
                    data={
                        "detail": "The CSV did not contain all the columns necessary to"
                        f" train the datamodel with id {datamodel.id}",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            return Response(
                data={"detail": "There is no datamodel with the specified id"},
                status=status.HTTP_404_NOT_FOUND,
            )

    @action(
        detail=True,
        methods=["POST"],
        url_path="train/finished",
        serializer_class=None,
        permission_classes=[],
    )
    def train_finished(self, request, *args, **kwargs):
        """Endpoint to get notified that the trainning of a model has ended."""
        datamodel = self.get_object()
        datamodel.set_trained()
        return Response(status=status.HTTP_200_OK)

    @action(
        detail=True, methods=["POST"], url_path="deploy", serializer_class=None,
    )
    def deploy(self, request, *args, **kwargs):
        """Deploys an Anomaly Detection Model."""

        datamodel = self.get_object()

        if datamodel.trained:
            datamodel.set_deployed()
            return Response(
                data={
                    "detail": f"The datamodel with id {datamodel.id} is now "
                    f"{'active' if datamodel.deployed else 'inactive'}"
                },
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                data={
                    "detail": f"The datatamodel with id {datamodel.id} is not trained. "
                    "Cannot deploy a datamodel which is not trained."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(detail=True, methods=["POST"], serializer_class=None, permission_classes=[])
    def predict(self, request, *args, **kwargs):
        """Receives new measures from Orion Context Broker subscriptions."""
        datamodel = self.get_object()

        if datamodel.trained and datamodel.deployed:
            data = request.data
            entity_data = data["data"][0]
            datamodel.set_subscription_data_and_predict(entity_data)
            return Response(status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)

    @action(
        detail=True,
        methods=["POST"],
        url_path="predict/result",
        serializer_class=None,
        permission_classes=[],
    )
    def predict_result(self, request, *args, **kwargs):
        """Receives prediction from Anomaly Detection API."""
        datamodel = self.get_object()
        datamodel.set_prediction_results(request.data)
        return Response(status=status.HTTP_200_OK)

    @action(
        detail=True,
        methods=["GET"],
        url_path="task_status",
        serializer_class=None,
        permission_classes=(permissions.IsAuthenticated,),
    )
    def task_status(self, request, *args, **kwargs):
        """Get the task status from a datamodel."""
        datamodel = self.get_object()
        task_status = datamodel.get_task_status()
        return Response(data=task_status, status=status.HTTP_200_OK)


class DatamodelPredictionViewSet(NestedViewSetMixin, viewsets.ReadOnlyModelViewSet):
    queryset = DataModelPrediction.objects.all()
    serializer_class = DataModelPredictionSerializer
    permission_classes = (permissions.IsAuthenticated,)

    @action(
        detail=True, methods=["POST"], url_path="ack",
    )
    def ack(self, request, *args, **kwargs):
        prediction = self.get_object()
        if not prediction.ack and prediction.user_ack and prediction.predictions != {}:
            prediction.set_ack(user=self.request.user.username)
            return Response(status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)
