# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/personality_types/package/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------
import requests
from pyspark.sql import SparkSession
import IPython
from personality_types.config import ProjectConfig

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = IPython.get_ipython().user_ns["dbutils"]
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
schema_path = f"{config.catalog_name}.{config.schema_name}"
test_set = (
    spark.table(f"{schema_path}.test_set")
    .drop("target", "update_timestamp_utc")
).toPandas()

sampled_records = test_set.sample(n=3, replace=False).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        .apiToken().get()
)
host = spark.conf.get("spark.databricks.workspaceUrl")
model_serving_endpoint = (
    f"https://{host}/serving-endpoints/personality-types-ab-test-model-serving/invocations"
)
# COMMAND ----------
response = requests.post(
    model_serving_endpoint,
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

print("Response status:", response.status_code)
print("Reponse text:", response.text)
