import mlflow
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig
from personality_types.test_model_wrapper import PersonalityTypesModelWrapper
from personality_types.utils.logger_utils import set_logger

mlflow.set_tracking_uri("databricks://adb-tuidiworkspace")
mlflow.set_registry_uri("databricks-uc://adb-tuidiworkspace")

spark = SparkSession.builder.getOrCreate()

logger = set_logger()

config = ProjectConfig.from_yaml(config_path="project_config.yml")

logger.info("Load models.")
schema_path = f"{config.catalog_name}.{config.schema_name}"
model_name = f"{schema_path}.personality_model_simple_test"

model_uri_a = f"models:/{model_name}@model_A"
model_uri_b = f"models:/{model_name}@model_B"

model_a = mlflow.sklearn.load_model(model_uri_a)
model_b = mlflow.sklearn.load_model(model_uri_b)

models = [model_a, model_b]

logger.info("Load train set.")
train_set_spark = spark.table(f"{schema_path}.train_set")

git_sha = "test"
run_tags = {"git_sha": git_sha, "branch": "week_3"}

model_wrapper = PersonalityTypesModelWrapper(models)

model_wrapper.log_model(
    spark,
    config,
    train_set_spark,
    "/Users/marco.dinardo@tuidi.it/personality-types-ab-testing",
    run_tags,
    "personality_types_model_pyfunc_ab_test",
)
