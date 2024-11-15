import mlflow
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from personality_types.config import ProjectConfig
from personality_types.personality_model import PersonalityModel
from personality_types.utils.logger_utils import set_logger

mlflow.set_tracking_uri("databricks://adb-tuidiworkspace")
mlflow.set_registry_uri("databricks-uc://adb-tuidiworkspace")

spark = SparkSession.builder.getOrCreate()

logger = set_logger()

config = ProjectConfig.from_yaml(config_path="./project_config.yml")
logger.info("Configuration loaded.")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.cat_features)
    ],
    remainder="passthrough",
)

# Model A
personality_model = PersonalityModel(preprocessor, config)

git_sha = "test"

run_tags = {"git_sha": git_sha, "branch": "week_2"}

model_version = personality_model.train_and_log(
    spark,
    "/Users/marco.dinardo@tuidi.it/personality-types-simple-test",
    run_tags,
    "personality_model_simple_test",
    "model_A",
)

# model B
config.parameters["n_estimators"] = 2
personality_model = PersonalityModel(preprocessor, config)

git_sha = "test"

run_tags = {"git_sha": git_sha, "branch": "week_2"}

model_version = personality_model.train_and_log(
    spark,
    "/Users/marco.dinardo@tuidi.it/personality-types-simple-test",
    run_tags,
    "personality_model_simple_test",
    "model_B",
)
