from delta.tables import DeltaTable
from pyspark.sql import SparkSession


def get_table_version(spark: SparkSession, table_path: str) -> int:
    delta_table = DeltaTable.forName(spark, table_path)
    delta_history = delta_table.history()
    return delta_history.orderBy("version", ascending=False).first()["version"]
