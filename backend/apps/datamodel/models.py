from django.db import models
from django.core.validators import int_list_validator


class DataModel(models.Model):
    """Class which holds everything related to a Blackbox Anomaly Detection model."""

    name = models.CharField(max_length=128, help_text="Model name")
    is_training = models.BooleanField(default=False)
    trained = models.BooleanField(default=False)
    deployed = models.BooleanField(default=False)
    date_trained = models.DateField(default=None, blank=True, null=True)
    date_deployed = models.DateField(default=None, blank=True, null=True)
    num_predictions = models.IntegerField(default=0)


class PCAMahanalobis(models.Model):
    """Class which holds the parameters of the PCA Mahalanobis anomaly detection model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
    n_components = models.IntegerField(
        default=2, help_text="Numbers of components for the PCA algorithm"
    )


class Autoencoder(models.Model):
    """Class which holds the parameters of the Autoencoder anomaly detection model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
    hidden_neurons = models.CharField(
        help_text="Neural Network layers and the number of neurons in each layer",
        validators=[int_list_validator],
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


class KMeans(models.Model):
    """Class which holds the parameters of the KMeans anomaly detection model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
    n_clusters = models.IntegerField(
        help_text="Number of clusters for the K-Means algorithm", null=True, blank=True,
    )
    max_cluster_elbow = models.IntegerField(
        help_text="Maximun number of cluster to test in the Elbow Method", default=100
    )


class OneClassSVM(models.Model):
    """Class which holds the parameters of the One Class SVM anomaly detection model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
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


class GaussianDistribution(models.Model):
    """Class which holds the parameters of the Gaussian Distribution anomaly detection
    model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
    epsilon_candidates = models.IntegerField(
        help_text="Number of epsilon values that will be tested to find the best one",
        default=100000000,
    )


class IsolationForest(models.Model):
    """Class which holds the parameters of the Isolation Forest anomaly detection model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
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


class LocalOutlierFactor(models.Model):
    """Class which holds the parameters of the Local Outlier Factor anomaly detection
    model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
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


class KNN(models.Model):
    """Class which holds the parameters of the Local Outlier Factor anomaly detection
    model."""

    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
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
