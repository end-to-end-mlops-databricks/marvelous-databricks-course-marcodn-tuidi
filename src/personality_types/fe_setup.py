from pyspark.sql import SparkSession
from src.personality_types.config import ProjectConfig


class TrainingSetBuilder:
    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark = spark
        self.config = config

    def create_features_table(self, table_name):
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        table = f"{schema_path}.{table_name}"
        self.spark.sql(
            f"""
            CREATE OR REPLACE TABLE {table}
            """
        )
