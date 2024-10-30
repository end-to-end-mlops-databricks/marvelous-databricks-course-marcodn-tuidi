from delta.tables import DeltaTable
from pyspark.sql import SparkSession


def get_table_version(spark: SparkSession, table_path: str) -> int:
    """
    Retrieves the current latest version of a Delta table.

    Args:
        spark (SparkSession): The active Spark session used to access the
            Delta table.
        table_path (str): The path or name of the Delta table for which to
            retrieve the latest version.

    Returns:
        int: The latest version number of the specified Delta table.
    """
    delta_table = DeltaTable.forName(spark, table_path)
    delta_history = delta_table.history()
    return delta_history.orderBy("version", ascending=False).first()["version"]
