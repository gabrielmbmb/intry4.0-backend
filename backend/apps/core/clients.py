import re
import logging
import requests
import pandas as pd
from crate import client as crate_client
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
        self.orion_host = orion_host or config.ORION_HOST
        self.orion_port = orion_port or config.ORION_PORT
        self.fiware_service = fiware_service or config.FIWARE_SERVICE
        self.fiware_servicepath = fiware_servicepath or config.FIWARE_SERVICEPATH

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
                f"Could not connect with Orion Context Broker at {self.orion_host}:{self.orion_port}: {e}"
            )
            raise OrionNotAvailable()

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
        self.blackbox_host = blackbox_host or config.BLACKBOX_HOST
        self.blackbox_port = blackbox_port or config.BLACKBOX_PORT

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
                    f"Could not create Blackbox {datamodel.id}. Anomaly Detection API is unavailable: {e}"
                )
                raise AnomalyDetectionNotAvailable()

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
                    f"Could not update Blackbox {datamodel.id}. Anomaly Detection API is unavailable: {e}"
                )
                raise AnomalyDetectionNotAvailable()

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
                f"Could not delete Blackbox {datamodel.id}. Anomaly Detection API is unavailable: {e}"
            )
            raise AnomalyDetectionNotAvailable()

        logger.info(f"Anomaly Detection has responded: {response.text}")

    def train(self, id, payload):
        """Trains a blackbox in the Anomaly Detection API.

        Args:
            id (:obj:`str`): id of the model which is going to be trained.
            payload (:obj:`str`): payload for training the blackbox model.

        Raises:
            AnomalyDetectionNotAvailable: if the Anomaly Detection is not available.
            AnomalyDetectionBadRequest: if a bad request has been made to the Anomaly
                Detection API.

        Returns:
            str: task status url.
        """
        url = f"http://{self.blackbox_host}:{self.blackbox_port}/api/v1/bb/models/{id}/train"

        try:
            logger.info(f"Training Blackbox {id} in Anomaly Detection API.")
            response = requests.post(url=url, json=payload)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(
                f"Could not train Blackbox {id}. Anomaly Detection API is unavailable: {e}"
            )
            raise AnomalyDetectionNotAvailable()

        if response.status_code != 202:
            logger.error(f"Could not train Blackbox with {id}: {response.text}")
            raise AnomalyDetectionBadRequest()

        logger.info(f"Anomaly Detection has responded: {response.text}")
        data_response = response.json()

        return data_response["task_status"]

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


class CrateClient(object):
    """A client class to connect with CrateDB

    Args:
        crate_host (:obj:`str`): CrateDB host. If the argument is not passed,
            the one from the Constance configuration will be taken. Defaults to None.
        crate_port (:obj:`str`): CrateDB port. If the argument is not passed,
            the one from the Constance configuration will be taken. Defaults to None.
        crate_db(:obj:`str`): name of the Crate database which holds all the tables. If
            the argument is not passed, the one from the Constance configuration will be
            taken. Defaults to None.

    Attributes:
        _connection (): internal attribute which holds the connection with CrateDB.
    """

    def __init__(
        self,
        crate_host: str = None,
        crate_port: str = None,
        crate_db: str = None,
        crate_metadata_table_name: str = None,
    ):
        self.crate_host = crate_host or config.CRATE_HOST
        self.crate_port = crate_port or config.CRATE_PORT
        self.crate_db = crate_db or config.CRATE_DB
        self._connection = None

    def _connect(self):
        """Connect with CrateDB."""
        if self._connection is None:
            logger.info(f"Connecting to Crate at {self.crate_host}:{self.crate_port}")
            url = f"{self.crate_host}:{self.crate_port}"
            self._connection = crate_client.connect(url)
            logger.info(f"Connected to Crate at {self.crate_host}:{self.crate_port}")
        else:
            logger.info("There is already a connection created to Crate")

    def _get_cursor(self):
        """Get a cursor from the connection stablished with Crate.

        Returns:
            crate.client.cursor.Cursor or None: the cursor. None if the connection with
                Crate is not stablished.
        """
        self._connect()

        if self._connection:
            return self._connection.cursor()
        else:
            return None

    def _execute_query(self, query: str):
        """Executes a query to CrateDB

        Args:
            query(:obj:`str`): the query.

        Returns:
            tuple: containing the rows and the columns info
        """
        cursor = self._get_cursor()

        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except crate_client.exceptions.ProgrammingError as e:
            logger.error(f"Crate query execute error: {e}")
            raise CrateNotAvailable()

        columns = [column[0] for column in cursor.description]

        return rows, columns

    @staticmethod
    def _get_entity_table_name(entity_urn: str) -> str:
        """Gets the table which holds the data of an entity identified with a URN

        Args:
            entity_urn (:obj:`str`): URN of the entity.

        Returns:
            str or None: name of the table which holds the data of the specified entity,
                or None if the entity URN is not valid.
        """

        # check wheter is a valid URN
        if not re.match(
            "^urn:[a-z0-9][a-z0-9-]{0,31}:[a-z0-9()+,\-.:=@;$_!*'%/?#]+$", entity_urn
        ):
            splitted = entity_urn.split(":")
            entity_type = splitted[2].lower() + str(int(splitted[3]))
            table_name = "et" + entity_type
            return table_name
        else:
            logger.error("Invalid URN")
            return None

    @staticmethod
    def _get_df(from_data: dict):
        """Gets a Pandas Dataframe from the dictionary with the results of a query to
        Crate. The dictionary contains keys which are the URN Of the entity. These keys
        contains another dictionary with two keys "rows" and "columns". The numbers of
        rows retreived from an entity can be less than another entity. Therefore, the
        resulting dataframe will have N-rows equal to the number of rows of the entity
        with less rows.

        Args:
            from_data (:obj:`dict`): data from a query to crate.

        Returns:
            pandas.core.frame.DataFrame: a dataframe with the data from the query.
        """

        dfs = []
        for entity in from_data.keys():
            df = pd.DataFrame(
                data=from_data[entity]["rows"], columns=from_data[entity]["columns"]
            )
            dfs.append(df)

        df = pd.concat(dfs, axis=1)
        df = df.dropna(axis=0)

        return df

    def get_data_from_plc(
        self,
        plc_sensors: dict,
        n: int = 100,
        from_date: str = None,
        to_date: str = None,
    ) -> dict:
        """Get historic data of PLCs.

        Args:
            plc_sensors (:obj:`dict`): in which the key is the URN of the PLC and the
                value is a list with the name of the sensors from which to obtain data.
            n (:obj:`int`): the number of rows to obtain. If specified, then `from_date`
                and `to_date` args should not be specified. Defaults to 100.
            from_date (:obj:`str`): date from which obtain data. Defaults to None.
            to_date (:obj:`str`): date until which obtain data. Defaults to None.

        Returns:
            pandas.core.frame.DataFrame: a dataframe with the data from the query.
        """

        # This dictionary will be used to translate back from "plc_table" to "plc_entity"
        plcs_table = {}

        # This one to make the queries
        plcs_to_query = {}

        for plc_entity in plc_sensors.keys():
            plc_table = self._get_entity_table_name(plc_entity)
            plcs_table[plc_table] = plc_entity
            plcs_to_query[plc_table] = plc_sensors[plc_entity]

        results = {}
        for (plc_table, plc_columns) in plcs_to_query.items():
            plc_columns_quoted = list(map(lambda x: f'"{x}"', plc_columns))

            if from_date and to_date:
                query = f"""
                    SELECT
                        {",".join(plc_columns_quoted)}
                    FROM "{self.crate_db}"."{plc_table}"
                    ORDER BY "time_index" ASC
                    WHERE "time_index" >= "
                """

            # query the first N rows
            else:
                query = f"""
                    SELECT
                        {",".join(plc_columns_quoted)}
                    FROM "{self.crate_db}"."{plc_table}"
                    ORDER BY "time_index" ASC
                    LIMIT {n}
                """
                logger.info(f"Executing query: {query}")

            rows, columns = self._execute_query(query)

            results[plcs_table[plc_table]] = {"rows": rows, "columns": columns}

        return self._get_df(results)


class CrateNotAvailable(APIException):
    """Raised when CrateDB is not available."""

    status_code = 504
    default_detail = "Unable to connect to Crate DB"
    default_code = "unable_to_connect_to_crate_db"
