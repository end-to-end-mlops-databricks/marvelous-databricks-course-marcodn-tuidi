# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/personality_types/package/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from personality_types.config import ProjectConfig
from personality_types.fe_setup import TrainingSetBuilder
from personality_types.personality_model import PersonalityModel
from personality_types.utils.logger_utils import set_logger

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()

logger = set_logger()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded.")

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.cat_features)
    ],
    remainder="passthrough",
)

# COMMAND ----------

logger.info("Create fe training set.")
git_sha = "test"
run_tags = {"git_sha": git_sha, "branch": "week_2"}

fe_builder = TrainingSetBuilder(spark, config)
training_set = fe_builder.create_training_set(fe)

# COMMAND ----------

personality_model = PersonalityModel(preprocessor, config)

model_version = personality_model.train_and_log_from_fe(
    spark,
    fe,
    "/Users/marco.dinardo@tuidi.it/personality-types-fe",
    run_tags,
    training_set,
)
