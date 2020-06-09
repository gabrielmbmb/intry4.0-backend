import logging
from django.db import models
from django.core.validators import (
    int_list_validator,
    MinValueValidator,
)

logger = logging.getLogger(__name__)


class DataModel(models.Model):
    """Class which holds everything related to a Blackbox Anomaly Detection model."""

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

    def save(self, *args, **kwargs):
        """Custom save method in order to create the model in the Anomaly Detection API
        before saving it."""
        logger.info("CREATING NEW MODEL")
        super(DataModel, self).save(*args, **kwargs)


class PLCModel(models.Model):
    """Class which holds everything related to a PLC and its sensors."""

    datamodel = models.ForeignKey(
        DataModel, related_name="plcs", on_delete=models.CASCADE
    )
    urn = models.CharField(
        help_text="Uniform Resource Name of the PLC in Orion Context Broker",
        max_length=256,
    )


class SensorModel(models.Model):
    """Class which holds everything related to a sensor."""

    plc = models.ForeignKey(PLCModel, related_name="sensors", on_delete=models.CASCADE)
    name = models.CharField(help_text="Name of the sensor", max_length=256)


class PCAMahalanobisModel(models.Model):
    """Class which holds the parameters of the PCA Mahalanobis anomaly detection model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
    n_components = models.IntegerField(
        help_text="Numbers of components for the PCA algorithm",
        default=2,
        validators=[MinValueValidator(1)],
    )


class AutoencoderModel(models.Model):
    """Class which holds the parameters of the Autoencoder anomaly detection model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
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
    )
    dropout_rate = models.FloatField(
        help_text="Dropout rate across all the layers of the Neural Network",
        default=0.2,
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
    )
    loss_function = models.CharField(
        help_text="Loss function of the Neural Network", default="mse", max_length=24
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
    )
    epochs = models.IntegerField(
        help_text="Number of times that all the batches will be processed in the "
        " Neural Network",
        default=100,
    )
    batch_size = models.IntegerField(help_text="Batch size", default=32)
    validation_split = models.FloatField(
        help_text="Percentage of the training data that will be used for purpouses in"
        " the Neural Network",
        default=0.05,
    )
    early_stopping = models.BooleanField(
        help_text="Stops the training process in the Neural Network when it's not"
        " getting any improvement",
        default=False,
    )


class KMeansModel(models.Model):
    """Class which holds the parameters of the KMeans anomaly detection model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
    n_clusters = models.IntegerField(
        help_text="Number of clusters for the K-Means algorithm", null=True, blank=True,
    )
    max_cluster_elbow = models.IntegerField(
        help_text="Maximun number of cluster to test in the Elbow Method", default=100
    )


class OneClassSVMModel(models.Model):
    """Class which holds the parameters of the One Class SVM anomaly detection model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
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
    )
    degree = models.IntegerField(
        help_text="Degree of the polynomal kernel function for One Class SVM", default=3
    )
    gamma = models.CharField(
        help_text="Kernel coefficient for 'rbf', 'poly' and 'sigmoid' in One Class SVM."
        " It can 'scale', 'auto' or float",
        default="scale",
        max_length=24,
    )
    coef0 = models.FloatField(
        help_text="Independent term in kernel function for One Class SVM. Only "
        "significant in 'poly'",
        default=0.0,
    )
    tol = models.FloatField(
        help_text="Tolerance for stopping criterion for One Class SVM", default=0.001
    )
    shrinking = models.BooleanField(
        help_text="Whether to use the shrinking heuristic for One Class SVM",
        default=True,
    )
    cache_size = models.IntegerField(
        help_text="Specify the size of the kernel cache in MB for One Class SVM",
        default=200,
    )


class GaussianDistributionModel(models.Model):
    """Class which holds the parameters of the Gaussian Distribution anomaly detection
    model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
    epsilon_candidates = models.IntegerField(
        help_text="Number of epsilon values that will be tested to find the best one",
        default=100000000,
    )


class IsolationForestModel(models.Model):
    """Class which holds the parameters of the Isolation Forest anomaly detection model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
    n_estimators = models.IntegerField(
        help_text="The number of base estimators in the ensemble for Isolation "
        "Forest",
        default=100,
    )
    max_features = models.FloatField(
        help_text="Number of features to draw from X to train each base estimator"
        " for Isolation Forest",
        default=1.0,
    )
    bootstrap = models.BooleanField(
        help_text="Indicates if the Bootstrap technique is going to be applied "
        "for Isolation FOrest",
        default=False,
    )


class LocalOutlierFactorModel(models.Model):
    """Class which holds the parameters of the Local Outlier Factor anomaly detection
    model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
    n_neighbors = models.IntegerField(
        help_text="Number of neighbors to use in LOF", default=20
    )
    algorithm = models.CharField(
        help_text="Algorithm used to compute the nearest neighbors in LOF",
        choices=[
            ("ball_tree", "ball_tree"),
            ("kd_tree", "kd_tree"),
            ("brute", "brute"),
            ("auto", "auto"),
        ],
        default="auto",
        max_length=24,
    )
    leaf_size = models.IntegerField(
        help_text="Leaf size passed to BallTree or KDTree in LOF", default=30
    )
    metric = models.CharField(
        help_text="The distance metric to use for the tree in LOF",
        default="minkowski",
        max_length=24,
    )
    p = models.IntegerField(
        help_text="Paremeter of the Minkowski metric in LOF", default=2
    )


class KNNModel(models.Model):
    """Class which holds the parameters of the Local Outlier Factor anomaly detection
    model."""

    datamodel = models.OneToOneField(DataModel, on_delete=models.CASCADE)
    n_neighbors = models.IntegerField(
        help_text="Number of neighbors to use in KNN", default=5
    )
    radius = models.FloatField(
        help_text="The range of parameter space to use by default for "
        "radius_neighbors",
        default=1.0,
    )
    algorithm = models.CharField(
        help_text="Algorithm used to compute the nearest neighbors in KNN",
        choices=[
            ("ball_tree", "ball_tree"),
            ("kd_tree", "kd_tree"),
            ("brute", "brute"),
            ("auto", "auto"),
        ],
        default="auto",
        max_length=24,
    )
    leaf_size = models.IntegerField(
        help_text="Leaf size passed to BallTree or KDTree in KNN", default=30,
    )
    metric = models.CharField(
        help_text="The distance metric to use for the tree in KNN",
        default="minkowski",
        max_length=24,
    )
    p = models.IntegerField(
        help_text="Paremeter of the Minkowski metric in knn", default=2,
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
    )
