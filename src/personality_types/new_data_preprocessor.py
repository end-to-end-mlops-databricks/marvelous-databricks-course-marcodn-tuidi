import time
from datetime import datetime

import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from pyspark.sql import DataFrame, SparkSession

from personality_types.config import ProjectConfig
from personality_types.utils.dbutils_utils import get_dbutils
from personality_types.utils.logger_utils import set_logger

logger = set_logger()


class NewDataProcessor:
    """
    A class responsible for updating the train and test set with new data.

    Attributes:
        config (ProjectConfig): A configuration object containing feature
            names and other processing parameters.
        workspace (WorkspaceClient): Databricks Workspace Client.
        latest_timestamp (datetime): latest processed update date of the
            source data. Everything after this date is considered new data.
    """

    def __init__(
        self,
        spark: SparkSession,
        workspace: WorkspaceClient,
        config: ProjectConfig,
        train_table_name: str,
        test_table_name: str,
    ) -> None:
        """
        Initializes the NewDataProcessor with the last update date.

        Args:
            spark (SparkSession): current spark session.
            workspace (WorkspaceClient): Databricks Workspace Client.
            config (dict): Configuration dictionary containing feature names
                and target variable details.
            train_table_name (str): Name of the train table.
            test_table_name (str): Name of the test table.
        """
        self.config = config
        self.workspace = workspace
        self.latest_timestamp = self.check_maximum_date(
            spark, train_table_name, test_table_name
        )

    def check_maximum_date(
        self,
        spark: SparkSession,
        train_table_name: str,
        test_table_name: str,
    ) -> datetime:
        """
        Check and return the most recent update timestamp across the train and
        test tables.

        Args:
            spark (SparkSession): The active Spark session for querying data.
            train_table_name (str): Name of the train table.
            test_table_name (str): Name of the test table.

        Returns:
            datetime: The latest update timestamp.
        """
        max_train_date = (
            self.load_delta(spark, train_table_name)
            .select(
                F.max("update_timestamp_utc").alias("max_update_timestamp")
            )
            .collect()[0]["max_update_timestamp"]
        )

        max_test_date = (
            self.load_delta(spark, test_table_name)
            .select(
                F.max("update_timestamp_utc").alias("max_update_timestamp")
            )
            .collect()[0]["max_update_timestamp"]
        )

        return max(max_train_date, max_test_date)

    def load_delta(self, spark: SparkSession, table_name: str) -> DataFrame:
        """
        Loads the dataset from the specified delta table in catalog.

        Args:
            spark (SparkSession): current spark session.
            table_name (str): Name of the table to load.

        Returns:
            DataFrame: Loaded dataset.
        """
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        return spark.table(f"{schema_path}.{table_name}")

    def load_new_data_to_train_test(
        self,
        spark: SparkSession,
        source_table_name: str,
        train_table_name: str,
        test_table_name: str,
        feature_table_name: str,
    ) -> None:
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        new_data = self.load_delta(spark, source_table_name).where(
            F.col("update_timestamp_utc") > self.latest_timestamp
        )
        logger.info("New data uploaded")
        new_data_train, new_data_test = new_data.randomSplit(
            [0.8, 0.2], seed=42
        )
        affected_rows_train = new_data_train.count()
        affected_rows_test = new_data_test.count()
        logger.info(f"New train rows: {affected_rows_train}")
        logger.info(f"New test rows: {affected_rows_test}")

        new_data_train.write.mode("append").saveAsTable(
            f"{schema_path}.{train_table_name}"
        )
        new_data_test.write.mode("append").saveAsTable(
            f"{schema_path}.{test_table_name}"
        )
        logger.info("Data added to train and test delta.")

        if affected_rows_train >= 0 or affected_rows_test >= 0:
            spark.sql(
                f"""
                WITH max_timestamp AS (
                    SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                    FROM {schema_path}.{train_table_name}
                )
                INSERT INTO {schema_path}.{feature_table_name}
                SELECT id, gender
                FROM {schema_path}.{train_table_name}
                WHERE update_timestamp_utc == (
                    SELECT max_update_timestamp FROM max_timestamp
                )
                """
            )

            spark.sql(
                f"""
                WITH max_timestamp AS (
                    SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                    FROM {schema_path}.{test_table_name}
                )
                INSERT INTO {schema_path}.{feature_table_name}
                SELECT id, gender
                FROM {schema_path}.{test_table_name}
                WHERE update_timestamp_utc == (
                    SELECT max_update_timestamp FROM max_timestamp
                )
                """
            )
            refreshed = 1
            update_response = self.workspace.pipelines.start_update(
                pipeline_id=self.config.pipeline_id,
                full_refresh=False,
            )
            logger.info("Feature table updated.")

            while True:
                update_info = self.workspace.pipelines.get_update(
                    pipeline_id=self.config.pipeline_id,
                    update_id=update_response.update_id,
                )
                state = update_info.update.state.value
                if state == "COMPLETED":
                    logger.info("Online table updated.")
                    break
                elif state in ["FAILED", "CANCELED"]:
                    raise SystemError("Online table failed to update.")
                elif state == "WAITING_FOR_RESOURCES":
                    logger.info("Pipeline is waiting for resources.")
                else:
                    logger.info(f"Pipeline is in {state} state.")
                time.sleep(30)
        else:
            refreshed = 1

        dbutils = get_dbutils(spark)
        dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)
