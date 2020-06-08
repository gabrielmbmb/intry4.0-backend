from rest_framework import serializers
from backend.apps.datamodel import models


class PCAMahanalobisSerializer(serializers.ModelSerializer):
    """Serializer class of the model `PCAMahanalobis`"""

    class Meta:
        model = models.PCAMahanalobis
        fields = ["n_components"]


class AutoencoderSerializer(serializers.ModelSerializer):
    """Serializer class of the model `Autoencoder`"""

    class Meta:
        model = models.Autoencoder
        fields = [
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
        ]


class KMeansSerializer(serializers.ModelSerializer):
    """Serializer class of the model `KMeans`"""

    class Meta:
        model = models.KMeans
        fields = ["n_clusters", "max_cluster_elbow"]


class OneClassSVMSerializer(serializers.ModelSerializer):
    """Serializer class of the model `OneClassSVM`"""

    class Meta:
        model = models.OneClassSVM
        fields = [
            "kernel",
            "degree",
            "gamma",
            "coef0",
            "tol",
            "shrinking",
            "cache_size",
        ]


class GaussianDistributionSerializer(serializers.ModelSerializer):
    """Serializer class of the model `GaussianDistribution`"""

    class Meta:
        model = models.GaussianDistribution
        fields = ["epsilon_candidates"]


class IsolationForestSerializer(serializers.ModelSerializer):
    """Serializer class of the model `IsolationForest`"""

    class Meta:
        model = models.IsolationForest
        fields = ["n_estimators", "max_features", "bootstrap"]


class LocalOutlierFactorSerializer(serializers.ModelSerializer):
    """Serializer class of the model `LocalOutlierFactor`"""

    class Meta:
        model = models.LocalOutlierFactor
        fields = ["n_neighbors", "algorithm", "leaf_size", "metric", "p"]


class KNNSerializer(serializers.ModelSerializer):
    """Serializer class of the model `KNN`"""

    class Meta:
        model = models.KNN
        fields = [
            "n_neighbors",
            "radius",
            "algorithm",
            "leaf_size",
            "metric",
            "p",
            "score_func",
        ]


class DataModelSerializer(serializers.ModelSerializer):
    """Serializer class of the model `DataModel`"""

    pca_mahalanobis = PCAMahanalobisSerializer(required=False)
    autoencoder = AutoencoderSerializer(required=False)
    kmeans = KMeansSerializer(required=False)
    one_class_svm = OneClassSVMSerializer(required=False)
    gaussian_distribution = GaussianDistributionSerializer(required=False)
    isolation_forest = IsolationForestSerializer(required=False)
    lof = LocalOutlierFactorSerializer(required=False)
    knn = KNNSerializer(required=False)

    class Meta:
        model = models.DataModel
        fields = [
            "id",
            "name",
            "is_training",
            "trained",
            "deployed",
            "date_trained",
            "date_deployed",
            "num_predictions",
            "pca_mahalanobis",
            "autoencoder",
            "kmeans",
            "one_class_svm",
            "gaussian_distribution",
            "isolation_forest",
            "lof",
            "knn",
        ]
        read_only_fields = [
            "is_training",
            "trained",
            "deployed",
            "date_trained",
            "date_deployed",
        ]

    # def validated_data():
    #     pass

    # def create(self, validated_data):
    #     pass
