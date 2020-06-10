from rest_framework import serializers
from backend.apps.datamodel import models


class DataModelSerializer(serializers.ModelSerializer):
    """Serializer class of the model `DataModel`"""

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
            "n_components",
            "autoencoder",
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
            "kmeans",
            "n_clusters",
            "max_cluster_elbow",
            "ocsvm",
            "kernel",
            "degree",
            "gamma",
            "coef0",
            "tol",
            "shrinking",
            "cache_size",
            "gaussian_distribution",
            "epsilon_candidates",
            "isolation_forest",
            "n_estimators",
            "max_features",
            "bootstrap",
            "lof",
            "n_neighbors_lof",
            "algorithm_lof",
            "leaf_size_lof",
            "metric_lof",
            "p_lof",
            "knn",
            "n_neighbors_knn",
            "radius",
            "algorithm_knn",
            "leaf_size_knn",
            "metric_knn",
            "p_knn",
            "score_func",
        )
        read_only_fields = (
            "is_training",
            "trained",
            "deployed",
            "date_trained",
            "date_deployed",
        )
