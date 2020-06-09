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

    def create_model(self):
        pass

    def update_model(self):
        pass

    def delete_model(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
