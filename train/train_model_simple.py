from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.personality_types.config import ProjectConfig
from src.personality_types.personality_model import PersonalityModel
from src.utils.logger_utils import set_logger

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

print(config.parameters)
personality_model = PersonalityModel(preprocessor, config)

git_sha = "test"

run_tags = {"git_sha": git_sha, "branch": "week_2"}

model_version = personality_model.train_and_log(
    spark,
    "/Users/marco.dinardo@tuidi.it/personality-types-simple",
    run_tags,
    "personality_model_simple",
)
