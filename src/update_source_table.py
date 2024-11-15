from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig
from personality_types.utils.logger_utils import set_logger
from personality_types.utils.source_table_update_utils import (
    create_synthetic_data,
    update_source,
)

spark = SparkSession.builder.getOrCreate()

logger = set_logger()

config = ProjectConfig.from_yaml(config_path="./project_config.yml")
logger.info("Configuration loaded.")

synthetic_data_df = create_synthetic_data(
    spark,
    config,
    "source_table",
    100,
)

update_source(
    spark,
    config,
    synthetic_data_df,
    "source_table",
)
