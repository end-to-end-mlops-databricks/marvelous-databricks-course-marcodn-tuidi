from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureEngineeringClient
from src.personality_types.config import ProjectConfig
from src.personality_types.data_processor import DataProcessor
from src.personality_types.fe_setup import TrainingSetBuilder
from src.personality_types.log_from_fe_dataset import fe_logger
from src.utils.logger_utils import set_logger

spark = DatabricksSession.builder.getOrCreate()
fe = FeatureEngineeringClient()

logger = set_logger()

volume_path = "/Volumes/marvelous_dev_ops/personality_types/data/"
file_name = "people_personality_types.csv"
data_path = volume_path + file_name

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded.")

# Initialize DataProcessor
data_processor = DataProcessor(data_path, True, config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
preprocessor = data_processor.preprocessor
logger.info("Data preprocessed.")

logger.info("Create fe training set.")

git_sha = "test"

run_tags = {"git_sha": git_sha, "branch": "week_2"}

fe_builder = TrainingSetBuilder(spark, config)

training_set = fe_builder.create_training_set(fe)

model_version = fe_logger(
    spark,
    config,
    fe,
    preprocessor,
    "/Users/marco.dinardo@tuidi.it/personality-types-fe",
    run_tags,
    training_set,
)
