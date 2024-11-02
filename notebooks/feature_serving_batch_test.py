# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/personality_types/package/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------
import requests
from pyspark.sql import SparkSession
import IPython

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = IPython.get_ipython().user_ns["dbutils"]

# COMMAND ----------
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------
serving_endpoint = f"https://{host}/serving-endpoints/personality-types-feature-serving/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"id": "42"}]},
)

print("Response status:", response.status_code)
print("Reponse text:", response.text)
