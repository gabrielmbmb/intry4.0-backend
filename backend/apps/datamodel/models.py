import uuid
import logging
from django.db import models
from django.core.validators import (
    int_list_validator,
    MinValueValidator,
)
from django.dispatch import receiver
from django.db.models.signals import pre_delete
from backend.apps.core import clients

logger = logging.getLogger(__name__)


class DataModel(models.Model):
    """Class which holds everything related to a Blackbox Anomaly Detection model."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=128, help_text="Model name")
    is_training = models.BooleanField(
        help_text="Wether the model is being trained or not", default=False,
    )
    trained = models.BooleanField(
        help_text="Wether the model is trained or not", default=False
    )
    deployed = models.BooleanField(
        help_text="Wether the model is deployed or not", default=False
    )
    date_trained = models.DateField(
        help_text="Date the model was trained", default=None, blank=True, null=True
    )
    date_deployed = models.DateField(
        help_text="Date the model was deployed", default=None, blank=True, null=True
    )
    num_predictions = models.IntegerField(
        help_text="Number of predictions made by this model", default=0
    )

    # sensors
    plcs = models.JSONField()

    # PCA Mahalanobis
    pca_mahalanobis = models.BooleanField(null=True, blank=True, default=False)
    n_components = models.IntegerField(
        help_text="Numbers of components for the PCA algorithm",
        default=2,
        validators=[MinValueValidator(1)],
        null=True,
        blank=True,
    )

    # Autoencoder
    autoencoder = models.BooleanField(null=True, blank=True, default=False)
    hidden_neurons = models.CharField(
        help_text="Neural Network layers and the number of neurons in each layer",
        validators=[
            int_list_validator(
                sep=",",
                message="It should be a string with a list of integers separeted by a comma",
                allow_negative=False,
            )
        ],
        default="32,16,16,32",
        max_length=128,
        null=True,
        blank=True,
    )
    dropout_rate = models.FloatField(
        help_text="Dropout rate across all the layers of the Neural Network",
        default=0.2,
        null=True,
        blank=True,
    )
    activation = models.CharField(
        help_text="Layers activation function of Neural Network",
        choices=[
            ("elu", "elu"),
            ("softmax", "softmax"),
            ("selu", "selu"),
            ("softplus", "softplus"),
            ("softsign", "softsign"),
            ("relu", "relu"),
            ("tanh", "tanh"),
            ("sigmoid", "sigmoid"),
            ("hard_sigmoid", "hard_sigmoid"),
            ("exponential", "exponential"),
        ],
        default="elu",
        max_length=24,
        null=True,
        blank=True,
    )
    kernel_initializer = models.CharField(
        help_text="Layers kernel initializer of Neural Network",
        choices=[
            ("Zeros", "Zeros"),
            ("Ones", "Ones"),
            ("Constant", "Constant"),
            ("RandomNormal", "RandomNormal"),
            ("RandomUniform", "RandomUniform"),
            ("TruncatedNormal", "TruncatedNormal"),
            ("VarianceScaling", "VarianceScaling"),
            ("Orthogonal", "Orthogonal"),
            ("Identity", "Identity"),
            ("lecun_uniform", "lecun_uniform"),
            ("glorot_normal", "glorot_normal"),
            ("glorot_uniform", "glorot_uniform"),
            ("he_normal", "he_normal"),
            ("lecun_normal", "lecun_normal"),
            ("he_uniform", "he_uniform"),
        ],
        default="glorot_uniform",
        max_length=24,
        null=True,
        blank=True,
    )
    loss_function = models.CharField(
        help_text="Loss function of the Neural Network",
        default="mse",
        max_length=24,
        null=True,
        blank=True,
    )
    optimizer = models.CharField(
        help_text="Optimizer of Neural Network",
        choices=[
            ("sgd", "sgd"),
            ("rmsprop", "rmsprop"),
            ("adagrad", "adagrad"),
            ("adadelta", "adadelta"),
            ("adam", "adam"),
            ("adamax", "adamax"),
            ("nadam", "nadam"),
        ],
        default="adam",
        max_length=24,
        null=True,
        blank=True,
    )
    epochs = models.IntegerField(
        help_text="Number of times that all the batches will be processed in the "
        " Neural Network",
        default=100,
        null=True,
        blank=True,
    )
    batch_size = models.IntegerField(
        help_text="Batch size", default=32, null=True, blank=True
    )
    validation_split = models.FloatField(
        help_text="Percentage of the training data that will be used for purpouses in"
        " the Neural Network",
        default=0.05,
        null=True,
        blank=True,
    )
    early_stopping = models.BooleanField(
        help_text="Stops the training process in the Neural Network when it's not"
        " getting any improvement",
        default=False,
        null=True,
        blank=True,
    )

    # K-Means
    kmeans = models.BooleanField(null=True, blank=True, default=False)
    n_clusters = models.IntegerField(
        help_text="Number of clusters for the K-Means algorithm", null=True, blank=True,
    )
    max_cluster_elbow = models.IntegerField(
        help_text="Maximun number of cluster to test in the Elbow Method",
        default=100,
        null=True,
        blank=True,
    )

    # One Class SVM
    ocsvm = models.BooleanField(null=True, blank=True, default=False)
    kernel = models.CharField(
        help_text="Kernel type for One Class SVM",
        choices=[
            ("linear", "linear"),
            ("poly", "poly"),
            ("rbf", "rbf"),
            ("sigmoid", "sigmoid"),
            ("precomputed", "precomputed"),
        ],
        default="rbf",
        max_length=24,
        null=True,
        blank=True,
    )
    degree = models.IntegerField(
        help_text="Degree of the polynomal kernel function for One Class SVM",
        default=3,
        null=True,
        blank=True,
    )
    gamma = models.CharField(
        help_text="Kernel coefficient for 'rbf', 'poly' and 'sigmoid' in One Class SVM."
        " It can 'scale', 'auto' or float",
        default="scale",
        max_length=24,
        null=True,
        blank=True,
    )
    coef0 = models.FloatField(
        help_text="Independent term in kernel function for One Class SVM. Only "
        "significant in 'poly'",
        default=0.0,
        null=True,
        blank=True,
    )
    tol = models.FloatField(
        help_text="Tolerance for stopping criterion for One Class SVM",
        default=0.001,
        null=True,
        blank=True,
    )
    shrinking = models.BooleanField(
        help_text="Whether to use the shrinking heuristic for One Class SVM",
        default=True,
        null=True,
        blank=True,
    )
    cache_size = models.IntegerField(
        help_text="Specify the size of the kernel cache in MB for One Class SVM",
        default=200,
        null=True,
        blank=True,
    )

    # Gaussian Distribution
    gaussian_distribution = models.BooleanField(null=True, blank=True, default=False)
    epsilon_candidates = models.IntegerField(
        help_text="Number of epsilon values that will be tested to find the best one",
        default=100000000,
        null=True,
        blank=True,
    )

    # Isolation Forest
    isolation_forest = models.BooleanField(null=True, blank=True, default=False)
    n_estimators = models.IntegerField(
        help_text="The number of base estimators in the ensemble for Isolation "
        "Forest",
        default=100,
        null=True,
        blank=True,
    )
    max_features = models.FloatField(
        help_text="Number of features to draw from X to train each base estimator"
        " for Isolation Forest",
        default=1.0,
        null=True,
        blank=True,
    )
    bootstrap = models.BooleanField(
        help_text="Indicates if the Bootstrap technique is going to be applied "
        "for Isolation FOrest",
        default=False,
        null=True,
        blank=True,
    )

    # Local Outlier Factor
    lof = models.BooleanField(null=True, blank=True, default=False)
    n_neighbors_lof = models.IntegerField(
        help_text="Number of neighbors to use in LOF", default=20, null=True, blank=True
    )
    algorithm_lof = models.CharField(
        help_text="Algorithm used to compute the nearest neighbors in LOF",
        choices=[
            ("ball_tree", "ball_tree"),
            ("kd_tree", "kd_tree"),
            ("brute", "brute"),
            ("auto", "auto"),
        ],
        default="auto",
        max_length=24,
        null=True,
        blank=True,
    )
    leaf_size_lof = models.IntegerField(
        help_text="Leaf size passed to BallTree or KDTree in LOF",
        default=30,
        null=True,
        blank=True,
    )
    metric_lof = models.CharField(
        help_text="The distance metric to use for the tree in LOF",
        default="minkowski",
        max_length=24,
        null=True,
        blank=True,
    )
    p_lof = models.IntegerField(
        help_text="Paremeter of the Minkowski metric in LOF",
        default=2,
        null=True,
        blank=True,
    )

    # K-Nearest Neighbors
    knn = models.BooleanField(null=True, blank=True, default=False)
    n_neighbors_knn = models.IntegerField(
        help_text="Number of neighbors to use in KNN", default=5, null=True, blank=True
    )
    radius = models.FloatField(
        help_text="The range of parameter space to use by default for "
        "radius_neighbors",
        default=1.0,
        null=True,
        blank=True,
    )
    algorithm_knn = models.CharField(
        help_text="Algorithm used to compute the nearest neighbors in KNN",
        choices=[
            ("ball_tree", "ball_tree"),
            ("kd_tree", "kd_tree"),
            ("brute", "brute"),
            ("auto", "auto"),
        ],
        default="auto",
        max_length=24,
        null=True,
        blank=True,
    )
    leaf_size_knn = models.IntegerField(
        help_text="Leaf size passed to BallTree or KDTree in KNN",
        default=30,
        null=True,
        blank=True,
    )
    metric_knn = models.CharField(
        help_text="The distance metric to use for the tree in KNN",
        default="minkowski",
        max_length=24,
        null=True,
        blank=True,
    )
    p_knn = models.IntegerField(
        help_text="Paremeter of the Minkowski metric in knn",
        default=2,
        null=True,
        blank=True,
    )
    score_func = models.CharField(
        help_text="The function used to score anomalies in KNN",
        choices=[
            ("max_distance", "max_distance"),
            ("average", "average"),
            ("median", "median"),
        ],
        default="max_distance",
        max_length=24,
        null=True,
        blank=True,
    )

    # clients
    blackbox_client = clients.BlackboxClient()
    created_in_blackbox = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        """Custom save method in order to create the model in the Anomaly Detection API
        before saving it."""
        print(self.created_in_blackbox)

        if not self.created_in_blackbox:
            self.created_in_blackbox = self.blackbox_client.create_blackbox(self)
        else:
            self.blackbox_client.update_blackbox(self)

        super(DataModel, self).save(*args, **kwargs)

    def get_models_columns(self):
        """Returns a dict containing two lists, one with the columns and the other
        with the models

        Returns:
            dict or None: containing two lists.
        """
        data = {"models": [], "columns": []}
        if self.pca_mahalanobis:
            data["models"].append("pca_mahalanobis")

        if self.autoencoder:
            data["models"].append("autoencoder")

        if self.kmeans:
            data["models"].append("kmeans")

        if self.ocsvm:
            data["models"].append("one_class_svm")

        if self.gaussian_distribution:
            data["models"].append("gaussian_distribution")

        if self.isolation_forest:
            data["models"].append("isolation_forest")

        if self.lof:
            data["models"].append("local_outlier_factor")

        if self.knn:
            data["models"].append("knearest_neighbors")

        for sensors in self.plcs.values():
            data["columns"] = data["columns"] + sensors

        if data["models"] and data["columns"]:
            return data

        return None


@receiver(pre_delete)
def pre_delete_datamodel_handler(sender, instance, **kwargs):
    """Handles the signal post delete of a model `DataModel` requesting Anomaly
    Detection to delete a Blackbox model

    Args:
        sender (backend.apps.models.DataModel): the datamodel just deleted.
    """
    instance.blackbox_client.delete_blackbox(sender)
