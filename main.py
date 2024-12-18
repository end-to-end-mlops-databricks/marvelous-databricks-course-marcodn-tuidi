from pyspark.sql import SparkSession
from src.personality_types.config import ProjectConfig
from src.personality_types.data_processor import DataProcessor
from src.utils.logger_utils import set_logger

spark = SparkSession.builder.getOrCreate()

logger = set_logger()

volume_path = "/Volumes/marvelous_dev_ops/personality_types/data/"
file_name = "people_personality_types.csv"
data_path = volume_path + file_name

config = ProjectConfig.from_yaml(config_path="project_config.yml")
logger.info("Configuration loaded.")

# Initialize DataProcessor
data_processor = DataProcessor(data_path, True, config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
preprocessor = data_processor.preprocessor
logger.info("Data preprocessed.")

# Split the data
data_processor.add_id_column()
X_train, X_test, y_train, y_test = data_processor.split_data()
logger.info("Data split into training and test sets.")

logger.info(
    f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}"
)

logger.info("Save train and test sets to catalog")
data_processor.save_to_catalog(X_train, X_test, y_train, y_test, spark)
