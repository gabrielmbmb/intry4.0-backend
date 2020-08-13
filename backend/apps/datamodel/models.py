import io
import uuid
import pytz
import json
import logging
import pandas as pd
from constance import config
from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.core.validators import (
    int_list_validator,
    MinValueValidator,
)
from django.db.models.signals import pre_delete
from datetime import datetime
from backend.apps.core import clients

logger = logging.getLogger(__name__)


NOT_ATTRIBUTES_KEYS_SUBSCRIPTION = ["id", "type", "TimeInstant"]


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
    date_trained = models.DateTimeField(
        help_text="Date the model was trained", default=None, blank=True, null=True
    )
    date_deployed = models.DateTimeField(
        help_text="Date the model was deployed", default=None, blank=True, null=True
    )
    num_predictions = models.IntegerField(
        help_text="Number of predictions made by this model", default=0
    )
    task_status = models.CharField(
        help_text="URL to get the progress of training process",
        null=True,
        blank=True,
        max_length=512,
    )

    # sensors
    plcs = JSONField()

    contamination = models.FloatField(
        help_text="Contamination fraction in the training dataset",
        default=0.1,
        validators=[MinValueValidator(0.0)],
        null=True,
        blank=True,
    )

    scaler = models.CharField(
        help_text="The scaler used to scale the data before training and predicting",
        default="minmax",
        max_length=48,
        null=True,
        blank=True,
    )

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
        help_text="Number of clusters for the K-Means algorithm",
        default=None,
        null=True,
        blank=True,
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

    # orion subscriptions
    subscriptions = ArrayField(models.CharField(max_length=128), default=list)

    # data from subscripitons
    data_from_subscriptions = JSONField(default=dict)
    dates = JSONField(default=dict)

    # clients
    blackbox_client = clients.BlackboxClient()
    crate_client = clients.CrateClient()
    orion_client = clients.OrionClient()

    def create_blackbox(self):
        """Creates a Blackbox model in the Anomaly Detection API."""
        self.blackbox_client.create_blackbox(self)

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

    def train(
        self,
        with_source: str,
        n: int = None,
        from_date: str = None,
        to_date: str = None,
        train_df=None,
    ) -> bool:
        """Trains the datamodel either with data from Crate or from a CSV

        Args:
            with_source (:obj:`str`): source of the training data. Valid choices are
                'db' or 'csv'.
            n (:obj:`int`): the number of rows to take from the database. Defaults to
                None.
            from_date (:obj:`str`): date from which the rows has to be taken. Defaults
                to None.
            to_date (:obj:`str`): date until which the rows has to be taken. Defaults to
                None.
            train_df (:obj:`pandas.core.frame.DataFrame`): the dataframe to perform the
                training of the model. Defaults to None.
        Returns:
            bool: wether the process of training has been initiated or not.
        """
        if not self.is_training:
            if with_source == "db":
                df = self.crate_client.get_data_from_plc(
                    self.plcs, n=n, from_date=from_date, to_date=to_date
                )

            # train with data from CSV
            else:
                df = train_df

            if df is None:
                return False

            train_data_json = json.loads(df.to_json(orient="split"))
            payload = self.to_json()
            payload["columns"] = train_data_json["columns"]
            payload["data"] = train_data_json["data"]

            self.task_status = self.blackbox_client.train(self.id, payload)
            self.is_training = True
            self.trained = False
            if self.deployed:
                self.set_deployed()
            self.save()

            return True

        return False

    def to_json(self):
        """Gets the model as json format."""
        json_ = {
            "contamination": self.contamination,
            "scaler": self.scaler,
            "n_jobs": -1,
            "pca_mahalanobis": {"n_components": self.n_components},
            "autoencoder": {
                "hidden_neurons": list(
                    map(lambda x: int(x), self.hidden_neurons.split(","))
                ),
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "loss_function": self.loss_function,
                "optimizer": self.optimizer,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "validation_split": self.validation_split,
                "early_stopping": self.early_stopping,
            },
            "kmeans": {"max_cluster_elbow": self.max_cluster_elbow},
            "one_class_svm": {
                "kernel": self.kernel,
                "degree": self.degree,
                "gamma": self.gamma,
                "coef0": self.coef0,
                "tol": self.tol,
                "shrinking": self.shrinking,
                "cache_size": self.cache_size,
            },
            "gaussian_distribution": {"epsilon_candidates": self.epsilon_candidates},
            "isolation_forest": {
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
                "bootstrap": self.bootstrap,
            },
            "knearest_neighbors": {
                "n_neighbors": self.n_neighbors_knn,
                "radius": self.radius,
                "algorithm": self.algorithm_knn,
                "leaf_size": self.leaf_size_knn,
                "metric": self.metric_knn,
                "p": self.p_knn,
                "score_func": self.score_func,
            },
            "local_outlier_factor": {
                "n_neighbors": self.n_neighbors_lof,
                "algorithm": self.algorithm_lof,
                "leaf_size": self.leaf_size_knn,
                "metric": self.metric_knn,
                "p": self.p_knn,
            },
        }

        if self.n_clusters:
            json_["kmeans"]["n_clusters"] = self.n_clusters

        return json_

    def set_trained(self):
        """Sets the datamodel to the trained state."""
        logger.info(f"Setting datamodel with id {self.id} to trained!")
        self.is_training = False
        self.trained = True
        self.date_trained = datetime.now(tz=pytz.UTC)
        self.save()

    def set_deployed(self):
        """Sets the datamodel to the deployed state."""
        self.deployed = not self.deployed

        if self.deployed:
            self.date_deployed = datetime.now(tz=pytz.UTC)

            # create subscriptions in OCB
            notification_url = (
                f"http://{config.SERVER_IP}/api/v1/datamodels/{self.id}/predict/"
            )

            subscriptions = []
            data_from_subscriptions = {}
            for (plc, sensors) in self.plcs.items():
                subscription = self.orion_client.create_subscription(
                    url=notification_url, pattern=plc, conditions=sensors, throttling=5
                )
                subscriptions.append(subscription)
                data_from_subscriptions[plc] = {}

            self.subscriptions = subscriptions
            self.data_from_subscriptions = data_from_subscriptions

        else:
            self.date_deployed = None

            # remove subscriptions in OCB
            self.orion_client.delete_subscriptions(self.subscriptions)
            self.subscriptions = []

        self.save()

    def check_csv_columns(self, file, index_column: str = None) -> bool:
        """Checks if a CSV has all the columns necessary to train this datamodel.

        Args:
            file (django.core.files.uploadedfile.TemporaryUploadedFile): training file.
            index_column (:obj:`str`): the name of the index column if there is one.
                Defaults to None.

        Returns:
            tuple: containing a bool which indicates if the CSV is valid. The second
                value is a dataframe in the case that CSV was valid or None if not.
        """
        if index_column:
            df = pd.read_csv(
                io.StringIO(file.read().decode("UTF-8")), index_col=index_column
            )
        else:
            df = pd.read_csv(io.StringIO(file.read().decode("UTF-8")))

        # get the columns that should be in the csv
        columns_that_should_be_in_csv = []
        for columns in self.plcs.values():
            for column in columns:
                columns_that_should_be_in_csv.append(column)

        columns_csv = list(df.columns)

        if all(
            column in columns_csv for column in columns_that_should_be_in_csv
        ) and all(column in columns_that_should_be_in_csv for column in columns_csv):
            return True, df

        return False, None

    def _all_data_from_subscriptions_received(self) -> bool:
        """Checks if data from all subscriptions has been received

        Returns:
            bool: weather if all data has been received.
        """
        return all(
            [data_sub != {} for data_sub in self.data_from_subscriptions.values()]
        )

    def _create_prediction_df(self):
        """Creates a dataframe which contains data from Orion subscriptions to make a
        prediction.

        Returns:
            pandas.core.frame.DataFrame: dataframe with data from subscriptions.
        """
        dfs = []
        data_from_subscriptions = {}
        for (plc, data_sub) in self.data_from_subscriptions.items():
            df = pd.DataFrame(data=data_sub["rows"], columns=data_sub["columns"])
            dfs.append(df)
            data_from_subscriptions[plc] = {}
        self.data_from_subscriptions = data_from_subscriptions
        df = pd.concat(dfs, axis=1)
        return df

    def set_subscription_data_and_predict(self, data: dict):
        """Sets subscription data and once it has received the data from all the
        subscriptions, it sends them to the Anomaly Detection API to generate a new
        prediction.

        Args:
            data (:obj:`str`): data from a subscription in OCB entity form.
        """
        entity_id = data["id"]

        # Get the attributes data of the subscription
        sub_data = {"rows": [[]], "columns": []}
        for key in data.keys():
            if key not in NOT_ATTRIBUTES_KEYS_SUBSCRIPTION:
                sub_data["rows"][0].append(data[key]["value"])
                sub_data["columns"].append(key)

        # save the data from this subscription
        if self.data_from_subscriptions[entity_id] == {}:
            # Save the time instant when the value of the sensors were updated
            for column in sub_data["columns"]:
                self.dates[column] = data["TimeInstant"]["value"]
            self.data_from_subscriptions[entity_id] = sub_data

        if self._all_data_from_subscriptions_received():
            df = self._create_prediction_df()
            payload = json.loads(df.to_json(orient="split"))
            prediction = DataModelPrediction(
                datamodel=self, data=payload.copy(), dates=self.dates
            )
            payload["id"] = str(prediction.id)
            prediction.task_status = self.blackbox_client.predict(self.id, payload)
            prediction.save()

        self.save()

    def send_prediction_to_orion(self, predictions: dict):
        """Sends the predictions received from the Anomaly Detection API to the Orion
        Context Broker.

        Args:
            predictions (:obj:`dict`): predictions made by the Anomaly Detection API.
        """
        prediction = DataModelPrediction.objects.get(
            datamodel=self, id=predictions["id"]
        )
        logger.debug(f"Prediction is: {prediction}")

        entity_id = f"urn:ngsi-ld:AnomalyPrediction:{self.id}"
        entity_type = "AnomalyPrediction"

        predictions_to_orion = {}

        for (key, value) in predictions.items():
            predictions_to_orion[key] = value[0]

        attrs = {
            "name": {"type": "String", "value": self.name},
            "entities": {"type": "Object", "value": self.plcs},
            "date": {"type": "DateTime", "value": datetime.now().isoformat()},
            "predictions": {"type": "Object", "value": predictions_to_orion},
        }

        self.orion_client.create_entity(entity_id, entity_type, attrs)
        self.num_predictions += 1
        self.save()

    def set_prediction_results(self, data: dict):
        """Set the results of the prediction received by the Anomaly Detection API.

        Args:
            data (:obj:`dict`): a dictionary containing the predictions and the ID of
                the prediction.
        """
        prediction = DataModelPrediction.objects.get(pk=data["id"])
        prediction.predictions = {
            key: value[0] for (key, value) in data.items() if key != "id"
        }
        prediction.save()
        self.num_predictions += 1
        self.save()
        prediction.send_to_orion()

    def get_task_status(self):
        """Gets the status of a task in the Anomaly Detection API."""
        return self.blackbox_client.get_task_status(self.task_status)


def pre_delete_datamodel_handler(sender, instance, **kwargs):
    """Handles the signal post delete of a model `DataModel` requesting Anomaly
    Detection to delete a Blackbox model

    Args:
        sender (backend.apps.models.DataModel): the datamodel just deleted.
    """
    instance.blackbox_client.delete_blackbox(instance)


pre_delete.connect(pre_delete_datamodel_handler, sender=DataModel)


class DataModelPrediction(models.Model):
    """Class which holds data of a prediction made by a `DataModel`."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
    data = JSONField()
    dates = JSONField()
    predictions = JSONField(default=dict)
    task_status = models.CharField(
        help_text="URL to get the progress of predicting process",
        null=True,
        blank=True,
        max_length=512,
    )
    ack = models.BooleanField(default=False)
    user_ack = models.CharField(max_length=128, blank=True, null=True)

    orion_client = clients.OrionClient()

    def send_to_orion(self):
        """Sends the prediction to the Orion Context Broker."""

        entity_id = f"urn:ngsi-ld:AnomalyPrediction:{self.id}"
        entity_type = "AnomalyPrediction"
        attrs = {
            "DataModelID": {"type": "String", "value": str(self.datamodel.id)},
            "DataModelName": {"type": "String", "value": self.datamodel.name},
            "Data": {"type": "Object", "value": self.data},
            "Dates": {"type": "Object", "value": self.dates},
            "Predictions": {"type": "Object", "value": self.predictions},
        }
        self.orion_client.create_entity(entity_id, entity_type, attrs)


class TrainFile(models.Model):
    datamodel = models.ForeignKey(DataModel, on_delete=models.CASCADE)
    file = models.FileField(
        blank=False,
        null=False,
        help_text="A CSV training file containing the columns of the DataModel",
    )
    index_column = models.CharField(max_length=128, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        get_latest_by = "uploaded_at"
