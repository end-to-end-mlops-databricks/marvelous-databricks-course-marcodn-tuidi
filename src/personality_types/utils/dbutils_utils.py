from typing import Any

from pyspark.sql import SparkSession


def get_dbutils(spark: SparkSession) -> Any:
    """
    Helper function to import dbutils.

    Args:
        spark (SparkSession): current spark session.
    Returns:
        Any: A dbutils instance.
    """
    dbutils = None

    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(spark)

    else:
        import IPython

        dbutils = IPython.get_ipython().user_ns["dbutils"]

    return dbutils
