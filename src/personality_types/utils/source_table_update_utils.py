import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig


def create_synthetic_data(
    spark: SparkSession,
    config: ProjectConfig,
    source_table_name: str,
    n_rows: int = 100,
) -> pd.DataFrame:
    """
    Util function to generate synthetic data to append to the source table.

    Args:
        spark (SparkSession): Current spark session.
        config (ProjectConfig): Configuration object with details for
            catalog, schema, and target variable.
        source_table_name (str): Name of the source table.
        n_rows (int): Number of rows to generate.

    Returns:
        pd.DataFrame: Synthetic dataset.
    """
    schema_path = f"{config.catalog_name}.{config.schema_name}"
    source_table_path = f"{schema_path}.{source_table_name}"
    source_df = spark.table(source_table_path)
    df = source_df.toPandas()
    synthetic_data = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != "id":
            mean, std = df[column].mean(), df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, n_rows)

        elif pd.api.types.is_categorical_dtype(
            df[column]
        ) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(),
                n_rows,
                p=df[column].value_counts(normalize=True),
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(
                    np.random.randint(min_date.value, max_date.value, n_rows)
                )
            else:
                synthetic_data[column] = [min_date] * n_rows

        else:
            synthetic_data[column] = np.random.choice(df[column], n_rows)

    existing_ids = set(int(id) for id in df["id"])
    start_id = max(existing_ids) + 1 if existing_ids else 1
    synthetic_data["id"] = [str(i) for i in range(start_id, start_id + n_rows)]

    synthetic_data_df = pd.DataFrame(synthetic_data)

    return synthetic_data_df


def update_source(
    spark: SparkSession,
    config: ProjectConfig,
    data: pd.DataFrame,
    source_table_name: str,
) -> None:
    """
    Util function to append data to the source table.

    Args:
        spark (SparkSession): Current spark session.
        config (ProjectConfig): Configuration object with details for
            catalog, schema, and target variable.
        data (pd.DataFrame): Dataframe we want to append to source table.
        source_table_name (str): Name of the source table.
    """
    schema_path = f"{config.catalog_name}.{config.schema_name}"
    source_table_path = f"{schema_path}.{source_table_name}"

    existing_schema = spark.table(source_table_path).schema
    data_spark = spark.createDataFrame(data, schema=existing_schema)

    data_spark = data_spark.withColumn(
        "update_timestamp_utc",
        F.to_utc_timestamp(F.current_timestamp(), "UTC"),
    )

    data_spark.write.mode("append").saveAsTable(source_table_path)
