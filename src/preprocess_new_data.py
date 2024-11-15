from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig
from personality_types.new_data_preprocessor import NewDataProcessor
from personality_types.utils.logger_utils import set_logger

spark = SparkSession.builder.getOrCreate()

logger = set_logger()

config = ProjectConfig.from_yaml(config_path="./project_config.yml")
logger.info("Configuration loaded.")

workspace = WorkspaceClient()

processor = NewDataProcessor(
    spark,
    workspace,
    config,
    "train_set",
    "test_set",
)

print(processor.latest_timestamp)
