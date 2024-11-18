import argparse

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig
from personality_types.models_comparator import Comparator
from personality_types.utils.dbutils_utils import get_dbutils
from personality_types.utils.logger_utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
new_model_uri = args.new_model_uri
job_run_id = args.job_run_id
git_sha = args.git_sha

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()
dbutils = get_dbutils(spark)
workspace = WorkspaceClient()

logger = set_logger()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded.")

run_tags = {
    "branch": "week_5",
    "git_sha": f"{git_sha}",
    "job_run_id": job_run_id,
}

comparator = Comparator(
    spark,
    workspace,
    config,
    "personality-types-fe-model-serving",
    new_model_uri,
    "test_set",
    "calculate_avg_score",
)

comparator.compare_models(fe, "personality_model_fe", run_tags)
