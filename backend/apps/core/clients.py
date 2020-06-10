import logging
import requests
from constance import config
from rest_framework.exceptions import APIException

logger = logging.getLogger(__name__)


class OrionClient(object):
    """A client class to connect with the Orion Context Broker.

    Args:
        orion_host (str): the Orion Context Broker host. If the argument is not passed,
            the one from the Constance configuration will be taken. Defaults to None.
        orion_port (str): the Orion Context Broker port. If the argument is not passed,
            the one from the Constance configuration will be taken. Defaults to None.
        fiware_service (str): the Orion Context Broker fiware service. If the argument
            is not passed, the one from the Constance configuration will be taken.
            Defaults to None.
        fiware_servicepath (str): the Orion Context Broker fiware service path. If the
            argument is not passed, the one from the Constance configuration will be
            taken. Defaults to None.
    """

    def __init__(
        self,
        orion_host: str = None,
        orion_port: str = None,
        fiware_service: str = None,
        fiware_servicepath: str = None,
    ):
        self.orion_host = orion_host if orion_host else config.ORION_HOST
        self.orion_port = orion_port if orion_port else config.ORION_PORT
        self.fiware_service = (
            fiware_service if fiware_service else config.FIWARE_SERVICE
        )
        self.fiware_servicepath = (
            fiware_servicepath if fiware_servicepath else config.FIWARE_SERVICEPATH
        )

    def get_entities(self, urn: str = None):
        """Gets an entity or all entities if urn is not passed from Orion Context Broker.

        Args:
            urn (str): the urn of the entity to get. Defaults to None.

        Raises:
            OrionNotAvailable: if Orion Context Broker is not available.

        Returns:
            dict: with the entity or entities data.
        """
        url = f"http://{self.orion_host}:{self.orion_port}/v2/entities"

        if urn is not None:
            url += f"/{urn}"

        headers = {
            "fiware-service": self.fiware_service,
            "fiware-servicepath": self.fiware_servicepath,
        }

        try:
            response = requests.get(url=url, headers=headers)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(
                f"Could not connect with Orion Context Broker at {self.orion_host}:{self.orion_port}"
            )
            raise OrionNotAvailable() from e

        return response.json()

    def create_subscription(self, pattern: str, url: str):
        pass


class OrionNotAvailable(APIException):
    """Raised when Orion Context Broker is not available."""

    status_code = 504
    default_detail = "Unable to connect to Orion Context Broker"
    default_code = "unable_to_connect_ocb"


class BlackboxClient(object):
    """A client class to connect with the Blackbox Anomaly Detection API

    Args:
        blackbox_host (str): the Blackbox host. If the argument is not passed,
            the one from the Constance configuration will be taken. Defaults to None.
        blackbox_port (str): the Blackbox port. If the argument is not passed,
            the one from the Constance configuration will be taken. Defaults to None.
    """

    def __init__(
        self, blackbox_host: str = None, blackbox_port: str = None,
    ):
        self.blackbox_host = blackbox_host if blackbox_host else config.BLACKBOX_HOST
        self.blackbox_port = blackbox_port if blackbox_port else config.BLACKBOX_PORT

    def create_blackbox(self, datamodel):
        """Create a blackbox in the Anomaly Detection API.

        Args:
            datamodel (backend.apps.models.DataModel): the datamodel with the related
                info of the blackbox.

        Raises:
            AnomalyDetectionNotAvailable: if the Anomaly Detection is not available.
            AnomalyDetectionBadRequest: if a bad request has been made to the Anomaly
                Detection API.

        Returns:
            bool: whether the Blackbox has been created or not.
        """

        data = datamodel.get_models_columns()

        if data:
            url = f"http://{self.blackbox_host}:{self.blackbox_port}/api/v1/bb/models/{datamodel.id}"
            payload = {"models": data["models"], "columns": data["columns"]}

            try:
                logger.info(
                    f"Creating Blackbox {datamodel.id} in Anomaly Detection API. Payload: {payload}"
                )
                response = requests.post(url=url, json=payload)
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.error(
                    f"Could not create Blackbox {datamodel.id}. Anomaly Detection API is unavailable..."
                )
                raise AnomalyDetectionNotAvailable() from e

            if response.status_code != 200:
                logger.error(
                    f"Could not create Blackbox {datamodel.id}: {response.text}"
                )
                raise AnomalyDetectionBadRequest()

            logger.info(f"Anomaly Detection has responded: {response.text}")
            return True

        return False

    def update_blackbox(self, datamodel):
        """Update a blackbox in the Anomaly Detection API.

        Args:
            datamodel (backend.apps.models.DataModel): the datamodel with the related
                info of the blackbox.

        Raises:
            AnomalyDetectionNotAvailable: if the Anomaly Detection is not available.
            AnomalyDetectionBadRequest: if a bad request has been made to the Anomaly
                Detection API.
        """

        data = datamodel.get_models_columns()

        if data:
            url = f"http://{self.blackbox_host}:{self.blackbox_port}/api/v1/bb/models/{datamodel.id}"
            payload = {"models": data["models"], "columns": data["columns"]}

            try:
                logger.info(
                    f"Updating Blackbox {datamodel.id} in Anomaly Detection API. Payload: {payload}"
                )
                response = requests.patch(url=url, json=payload)
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.error(
                    f"Could not update Blackbox {datamodel.id}. Anomaly Detection API is unavailable..."
                )
                raise AnomalyDetectionNotAvailable() from e

            if response.status_code != 200:
                logger.error(
                    f"Could not update Blackbox {datamodel.id}: {response.text}"
                )
                raise AnomalyDetectionBadRequest()

            logger.info(f"Anomaly Detection has responded: {response.text}")

    def delete_blackbox(self, datamodel):
        """Delete a blackbox in the Anomaly Detection API.

        Args:
            datamodel (backend.apps.models.DataModel): the datamodel with the related
                info of the blackbox.

        Raises:
            AnomalyDetectionNotAvailable: if the Anomaly Detection is not available.
            AnomalyDetectionBadRequest: if a bad request has been made to the Anomaly
                Detection API.
        """

        url = f"http://{self.blackbox_host}:{self.blackbox_port}/api/v1/bb/models/{datamodel.id}"

        try:
            logger.info(f"Deleting Blackbox {datamodel.id} in Anomaly Detection API.")
            response = requests.delete(url=url)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(
                f"Could not delete Blackbox {datamodel.id}. Anomaly Detection API is unavailable..."
            )
            raise AnomalyDetectionNotAvailable() from e

        logger.info(f"Anomaly Detection has responded: {response.text}")

    def train(self):
        pass

    def predict(self):
        pass


class AnomalyDetectionNotAvailable(APIException):
    """Raised when Anomaly Detection is not available."""

    status_code = 504
    default_detail = "Unable to connect to Anomaly Detection API"
    default_code = "unable_to_connect_anomaly_detection_api"


class AnomalyDetectionBadRequest(APIException):
    """Raised when a bad request has been made to the Anomaly Detection API."""

    status_code = 400
    default_datail = "Bad request to Anomaly Detection API"
    default_code = "bad_request_anomaly_detection_api"
