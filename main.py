import logging

import yaml
from src.personality_types.data_processor import DataProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

volume_path = "/Volumes/marvelous_dev_ops/personality_types/data/"
file_name = "people_personality_types.csv"
data_path = volume_path + file_name

# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# Initialize DataProcessor
data_processor = DataProcessor(data_path, True, config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
preprocessor = data_processor.preprocessor
logger.info("Data preprocessed.")

# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()
logger.info("Data split into training and test sets.")

logger.debug(
    f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}"
)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.fit_transform(X_test)
logger.info("Data processed.")

print(
    f"""Processed training set shape: {X_train_preprocessed.shape} Processed
    test set shape: {X_test_preprocessed.shape}"""
)
logger.debug(
    f"""Processed training set shape: {X_train_preprocessed.shape} Processed
    test set shape: {X_test_preprocessed.shape}"""
)
