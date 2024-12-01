from typing import Any, Dict

import mlflow
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import DataFrame, SparkSession

from personality_types.config import ProjectConfig
from personality_types.utils.dbutils_utils import get_dbutils
from personality_types.utils.logger_utils import set_logger

logger = set_logger()


class Comparator:
    def __init__(
        self,
        spark: SparkSession,
        workspace: WorkspaceClient,
        config: ProjectConfig,
        endpoint_name: str,
        new_model_uri: str,
        test_table_name: str,
        udf_name: str,
    ) -> None:
        self.spark = spark
        self.config = config
        self.old_model_uri = self.get_old_model_uri(workspace, endpoint_name)
        self.new_model_uri = new_model_uri
        self.test_set = self.load_test_set(test_table_name, udf_name)
        self.X_test = self.test_set.drop(self.config.target)
        self.y_test = self.test_set.select("id", self.config.target)

    def get_old_model_uri(
        self, workspace: WorkspaceClient, endpoint_name: str
    ) -> str:
        serving_endpoint = workspace.serving_endpoints.get(endpoint_name)
        model_name = serving_endpoint.config.served_models[0].model_name
        model_version = serving_endpoint.config.served_models[0].model_version
        return f"models:/{model_name}/{model_version}"

    def load_test_set(self, test_table_name: str, udf_name: str) -> DataFrame:
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        test_set = (
            self.spark.table(f"{schema_path}.{test_table_name}")
            .drop("update_timestamp_utc", "gender")
            .withColumn(
                "score_avg",
                F.expr(
                    f"{schema_path}.{udf_name}(thinking_score, sensing_score)"
                ),
            )
        )
        return test_set

    def compare_models(
        self,
        fe: FeatureEngineeringClient,
        model_name: str,
        run_tags: Dict[str, Any],
    ) -> None:
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"

        predictions_old = fe.score_batch(
            model_uri=self.old_model_uri, df=self.X_test, result_type="string"
        )
        predictions_new = fe.score_batch(
            model_uri=self.new_model_uri, df=self.X_test, result_type="string"
        )

        predictions_old = predictions_old.withColumnRenamed(
            "prediction", "prediction_old"
        )
        predictions_new = predictions_new.withColumnRenamed(
            "prediction", "prediction_new"
        )

        compare_df = (
            self.y_test.join(predictions_old, on="id", how="inner")
            .join(predictions_new, on="id", how="inner")
            .withColumn(
                "error_old",
                F.when(
                    F.col(self.config.target) == F.col("prediction_old"),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .withColumn(
                "error_new",
                F.when(
                    F.col(self.config.target) == F.col("prediction_new"),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .agg(
                F.mean(F.col("error_old")).alias("acc_old"),
                F.mean(F.col("error_new")).alias("acc_new"),
            )
        )

        print(compare_df.toPandas().head())

        acc_old = compare_df.select("acc_old").collect()[0][0]
        acc_new = compare_df.select("acc_new").collect()[0][0]

        logger.info(f"Old accuracy: {acc_old}")
        logger.info(f"New accuracy: {acc_new}")
        dbutils = get_dbutils(self.spark)
        if acc_new > acc_old:
            logger.info("New model is better than old model")

            mlflow.set_registry_uri("databricks-uc")
            mlflow.set_tracking_uri("databricks")

            model_version = mlflow.register_model(
                model_uri=self.new_model_uri,
                name=f"{schema_path}.{model_name}",
                tags=run_tags,
            )

            logger.info("New mdodel registered")
            dbutils.jobs.taskValues.set(
                key="model_version",
                value=model_version.version,
            )
            dbutils.jobs.taskValues.set(key="model_update", value=1)
        else:
            logger.info("Old model is better than new model")
            dbutils.jobs.taskValues.set(key="model_update", value=0)
