from typing import Any

from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureFunction,
    FeatureLookup,
)
from pyspark.sql import SparkSession
from src.personality_types.config import ProjectConfig


class TrainingSetBuilder:
    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark = spark
        self.config = config

    def create_lookup_table(self):
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        table = f"{schema_path}.personality_features"
        self.spark.sql(
            f"""
            CREATE OR REPLACE TABLE {table}
            (id STRING NOT NULL,
            gender STRING);
            """
        )

        self.spark.sql(
            f"""
            ALTER TABLE {table}
            ADD CONSTRAINT personality_pk PRIMARY KEY(id);
            """
        )

        self.spark.sql(
            f"""
            ALTER TABLE {table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
            """
        )

        self.spark.sql(
            f"""
            INSERT INTO {table}
            SELECT id, gender FROM {schema_path}.train_set
            """
        )

        self.spark.sql(
            f"""
            INSERT INTO {table}
            SELECT id, gender FROM {schema_path}.test_set
            """
        )

    def create_udf(self):
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"

        self.spark.sql(
            f"""
            CREATE OR REPLACE FUNCTION {schema_path}.calculate_avg_score(
                thinking_score DOUBLE,
                sensing_score DOUBLE
            )
            RETURNS INT
            LANGUAGE PYTHON AS
            $$
            return int((thinking_score + sensing_score) / 2)
            $$
            """
        )

    def create_training_set(self, fe: FeatureEngineeringClient) -> Any:
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_set = self.spark.table(f"{schema_path}.train_set").drop("gender")

        self.create_lookup_table()
        self.create_udf()

        training_set = fe.create_training_set(
            df=train_set,
            label=self.config.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=f"{schema_path}.personality_features",
                    feature_names=["gender"],
                    lookup_key="id",
                ),
                FeatureFunction(
                    udf_name=f"{schema_path}.calculate_avg_score",
                    output_name="score_avg",
                    input_bindings={
                        "thinking_score": "thinking_score",
                        "sensing_score": "sensing_score",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        return training_set
