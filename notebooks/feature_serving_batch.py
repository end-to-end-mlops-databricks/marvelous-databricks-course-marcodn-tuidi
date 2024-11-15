# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/personality_types/package/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import mlflow
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from personality_types.config import ProjectConfig
from personality_types.utils.logger_utils import set_logger
from personality_types.personality_model import PersonalityModel

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

logger = set_logger()

spark = SparkSession.builder.getOrCreate()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
schema_path = f"{config.catalog_name}.{config.schema_name}"

# COMMAND ----------
logger.info("Load dataset.")
train_table_path = f"{schema_path}.train_set"
test_table_path = f"{schema_path}.test_set"

df = (
    spark.table(train_table_path)
    .unionByName(
        spark.table(test_table_path)
    )
).toPandas()

# COMMAND ----------
model_name = "personality_model_basic"
model_version = 8

model_path = f"models:/{schema_path}.{model_name}/{model_version}"
model = PersonalityModel(
    preprocessor=None,
    config=config,
    model_path=model_path
)

# COMMAND ----------
df_predictions_spark = model.batch_predictions(
    spark,
    df
)

# COMMAND ----------
model.create_bach_feature_serving(
    spark,
    workspace,
    fe,
    df_predictions_spark,
    "personality_types_preds",
    "personality_types_preds_online",
    "return_predictions",
    "personality-types-feature-serving"
)
