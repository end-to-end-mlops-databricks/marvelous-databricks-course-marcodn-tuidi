import argparse

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig
from personality_types.new_data_preprocessor import NewDataProcessor
from personality_types.utils.logger_utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
args = parser.parse_args()

spark = SparkSession.builder.getOrCreate()

logger = set_logger()

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded.")

workspace = WorkspaceClient()

processor = NewDataProcessor(
    spark,
    workspace,
    config,
    "train_set",
    "test_set",
)

logger.info(f"Last processed maximum date: {processor.latest_timestamp}")

processor.load_new_data_to_train_test(
    spark,
    "source_table",
    "train_set",
    "test_set",
    "personality_features",
)
