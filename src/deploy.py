import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig
from personality_types.utils.dbutils_utils import get_dbutils
from personality_types.utils.logger_utils import set_logger

logger = set_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded.")

spark = SparkSession.builder.getOrCreate()
dbutils = get_dbutils(spark)


model_version = dbutils.jobs.taskValues.get(
    taskKey="evaluate_model", key="model_version"
)

workspace = WorkspaceClient()
schema_path = f"{config.catalog_name}.{config.schema_name}"

workspace.serving_endpoints.update_config_and_wait(
    name="house-prices-model-serving-fe",
    served_entities=[
        ServedEntityInput(
            entity_name=f"{schema_path}.personality_model_fe",
            scale_to_zero_enabled=True,
            workload_size="Small",
            entity_version=model_version,
        ),
    ],
)
