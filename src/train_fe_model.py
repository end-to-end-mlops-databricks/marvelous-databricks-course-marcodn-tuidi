import argparse

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from personality_types.config import ProjectConfig
from personality_types.fe_setup import TrainingSetBuilder
from personality_types.personality_model import PersonalityModel
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
    "--git_sha",
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

args = parser.parse_args()
git_sha = args.git_sha
job_run_id = args.job_run_id

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()
dbutils = get_dbutils(spark)

logger = set_logger()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded.")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.cat_features)
    ],
    remainder="passthrough",
)
logger.info("Preprocessor initialized")


git_sha = "test"
run_tags = {
    "branch": "week_5",
    "git_sha": f"{git_sha}",
    "job_run_id": job_run_id,
}

fe_builder = TrainingSetBuilder(spark, config)
training_set = fe_builder.create_training_set(fe)
logger.info("Fe training set created.")

personality_model = PersonalityModel(preprocessor, config)

model_version = personality_model.train_and_log_from_fe(
    spark,
    fe,
    "/Users/marco.dinardo@tuidi.it/personality-types-fe",
    run_tags,
    training_set,
    register_model=False,
)

dbutils.jobs.taskValues.set(
    key="new_model_uri", value=personality_model.model_uri
)
