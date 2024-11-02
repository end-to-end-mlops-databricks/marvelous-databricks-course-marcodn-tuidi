from pyspark.sql import SparkSession
from src.personality_types.config import ProjectConfig
from src.personality_types.data_processor import DataProcessor
from src.personality_types.personality_model import PersonalityModel
from src.personality_types.utils.logger_utils import set_logger

spark = SparkSession.builder.getOrCreate()

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

print(config.parameters)
personality_model = PersonalityModel(preprocessor, config)

git_sha = "test"

run_tags = {"git_sha": git_sha, "branch": "week_2"}

model_version = personality_model.train_and_log(
    spark,
    "/Users/marco.dinardo@tuidi.it/personality-types",
    run_tags,
)
