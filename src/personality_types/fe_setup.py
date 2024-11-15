from typing import Any

from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureFunction,
    FeatureLookup,
)
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig


class TrainingSetBuilder:
    """
    A class to build a training dataset for personality classification by
    creating lookup tables, defining user-defined functions (UDFs), and
    leveraging the Databricks Feature Engineering Client.

    Attributes:
        spark (SparkSession): The active Spark session for executing SQL
            queries and creating tables.
        config (ProjectConfig): Configuration object containing catalog,
            schema, and target variable information.
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig):
        """
        Initializes the TrainingSetBuilder with a Spark session and
        configuration.

        Args:
            spark (SparkSession): The active Spark session for executing SQL
                queries and interacting with Databricks.
            config (ProjectConfig): Configuration object with details for
                catalog, schema, and target variable.
        """
        self.spark = spark
        self.config = config

    def create_lookup_table(self):
        """
        Creates or replaces a lookup table named `personality_features` in
        the configured database schema. Adds a primary key constraint on
        the `id` column, enables change data feed for tracking changes, and
        populates the table with `id` and `gender` data from the training
        and test datasets.
        """
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
        """
        Creates or replaces a Python-based UDF named `calculate_avg_score` in
        the configured schema. This UDF calculates an average score from
        `thinking_score` and `sensing_score` by returning their mean value.
        """
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
        """
        Builds and returns a training set by integrating data transformations
        and feature lookups using the Feature Engineering Client.

        This method first creates the lookup table and UDF, then uses the
        Feature Engineering Client to construct a training set with `gender`
        as a lookup feature and `score_avg` as a calculated feature.

        Args:
            fe (FeatureEngineeringClient): Client for feature engineering to
                manage feature lookups and transformations.

        Returns:
            Any: A training set object from the Feature Engineering Client
            containing features, label, and additional transformations.
        """
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
