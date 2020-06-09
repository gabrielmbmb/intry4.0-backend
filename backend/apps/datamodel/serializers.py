from rest_framework import serializers
from backend.apps.datamodel import models


class PCAMahalanobisSerializer(serializers.ModelSerializer):
    """Serializer class of the model `PCAMahalanobisModel`"""

    class Meta:
        model = models.PCAMahalanobisModel
        fields = ("n_components",)


class AutoencoderSerializer(serializers.ModelSerializer):
    """Serializer class of the model `AutoencoderModel`"""

    class Meta:
        model = models.AutoencoderModel
        fields = (
            "hidden_neurons",
            "dropout_rate",
            "activation",
            "kernel_initializer",
            "loss_function",
            "optimizer",
            "epochs",
            "batch_size",
            "validation_split",
            "early_stopping",
        )


class KMeansSerializer(serializers.ModelSerializer):
    """Serializer class of the model `KMeansModel`"""

    class Meta:
        model = models.KMeansModel
        fields = (
            "n_clusters",
            "max_cluster_elbow",
        )


class OneClassSVMSerializer(serializers.ModelSerializer):
    """Serializer class of the model `OneClassSVMModel`"""

    class Meta:
        model = models.OneClassSVMModel
        fields = (
            "kernel",
            "degree",
            "gamma",
            "coef0",
            "tol",
            "shrinking",
            "cache_size",
        )


class GaussianDistributionSerializer(serializers.ModelSerializer):
    """Serializer class of the model `GaussianDistributionModel`"""

    class Meta:
        model = models.GaussianDistributionModel
        fields = ("epsilon_candidates",)


class IsolationForestSerializer(serializers.ModelSerializer):
    """Serializer class of the model `IsolationForestModel`"""

    class Meta:
        model = models.IsolationForestModel
        fields = (
            "n_estimators",
            "max_features",
            "bootstrap",
        )


class LocalOutlierFactorSerializer(serializers.ModelSerializer):
    """Serializer class of the model `LocalOutlierFactorModel`"""

    class Meta:
        model = models.LocalOutlierFactorModel
        fields = (
            "n_neighbors",
            "algorithm",
            "leaf_size",
            "metric",
            "p",
        )


class KNNSerializer(serializers.ModelSerializer):
    """Serializer class of the model `KNNModel`"""

    class Meta:
        model = models.KNNModel
        fields = (
            "n_neighbors",
            "radius",
            "algorithm",
            "leaf_size",
            "metric",
            "p",
            "score_func",
        )


class SensorSerializer(serializers.ModelSerializer):
    """Serializer class of the model `SensorModel`"""

    class Meta:
        model = models.SensorModel
        fields = ("name",)


class PLCSerializer(serializers.ModelSerializer):
    """Serializer class of the model `PLCModel`"""

    sensors = SensorSerializer(many=True)

    class Meta:
        model = models.PLCModel
        fields = (
            "urn",
            "sensors",
        )


class DataModelSerializer(serializers.ModelSerializer):
    """Serializer class of the model `DataModel`"""

    plcs = PLCSerializer(many=True, required=True)
    pca_mahalanobis = PCAMahalanobisSerializer(
        required=False, source="pcamahalanobismodel"
    )
    autoencoder = AutoencoderSerializer(required=False, source="autoencodermodel")
    kmeans = KMeansSerializer(required=False, source="kmeansmodel")
    one_class_svm = OneClassSVMSerializer(required=False, source="oneclasssvmmodel")
    gaussian_distribution = GaussianDistributionSerializer(
        required=False, source="gaussiandistributionmodel"
    )
    isolation_forest = IsolationForestSerializer(
        required=False, source="isolationforestmodel"
    )
    lof = LocalOutlierFactorSerializer(required=False, source="lofmodel")
    knn = KNNSerializer(required=False, source="knnmodel")

    class Meta:
        model = models.DataModel
        fields = (
            "id",
            "name",
            "is_training",
            "trained",
            "deployed",
            "date_trained",
            "date_deployed",
            "num_predictions",
            "plcs",
            "pca_mahalanobis",
            "autoencoder",
            "kmeans",
            "one_class_svm",
            "gaussian_distribution",
            "isolation_forest",
            "lof",
            "knn",
        )
        read_only_fields = (
            "is_training",
            "trained",
            "deployed",
            "date_trained",
            "date_deployed",
        )

    def validate(self, data):
        # check if at least one model has been provided
        MODELS = [
            "pcamahalanobismodel",
            "autoencodermodel",
            "kmeansmodel",
            "oneclasssvmmodel",
            "gaussian_distributionmodel",
            "isolation_forestmodel",
            "lofmodel",
            "knnmodel",
        ]

        if not any(model in data.keys() for model in MODELS):
            raise serializers.ValidationError(
                f"At least one model should be provided. Models available: {MODELS}"
            )

        return data

    def create(self, validated_data):
        plcs = validated_data.pop("plcs")
        pca_mahalanobis = validated_data.pop("pcamahalanobismodel", None)
        autoencoder = validated_data.pop("autoencodermodel", None)
        kmeans = validated_data.pop("kmeansmodel", None)
        one_class_svm = validated_data.pop("oneclasssvmmodel", None)
        gaussian_distribution = validated_data.pop("gaussiandistributionmodel", None)
        isolation_forest = validated_data.pop("isolationforestmodel", None)
        lof = validated_data.pop("lofmodel", None)
        knn = validated_data.pop("knnmodel", None)

        datamodel = models.DataModel.objects.create(**validated_data)

        # create PLCs and its Sensors
        for plc_data in plcs:
            sensors = plc_data.pop("sensors")
            plc = models.PLCModel.objects.create(datamodel=datamodel, **plc_data)

            for sensor in sensors:
                models.SensorModel.objects.create(plc=plc, **sensor)

        # create anomaly detection models
        if pca_mahalanobis is not None:
            models.PCAMahalanobisModel.objects.create(
                datamodel=datamodel, **pca_mahalanobis
            )

        if autoencoder is not None:
            models.AutoencoderModel.objects.create(datamodel=datamodel, **autoencoder)

        if kmeans is not None:
            models.KMeansModel.objects.create(datamodel=datamodel, **kmeans)

        if one_class_svm is not None:
            models.OneClassSVMModel.objects.create(datamodel=datamodel, **one_class_svm)

        if gaussian_distribution is not None:
            models.GaussianDistributionModel.objects.create(
                datamodel=datamodel, **gaussian_distribution
            )

        if isolation_forest is not None:
            models.IsolationForestModel.objects.create(
                datamodel=datamodel, **isolation_forest
            )

        if lof is not None:
            models.LocalOutlierFactorModel.objects.create(datamodel=datamodel, **lof)

        if knn is not None:
            models.KNNModel.objects.create(datamodel=datamodel, **knn)

        return datamodel
