# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/personality_types/package/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from personality_types.config import ProjectConfig
from personality_types.registered_model_serving import create_model_serving

# COMMAND ----------
workspace = WorkspaceClient()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
create_model_serving(
    config=config,
    workspace=workspace,
    endpoint_name="personality-types-simple-model-serving",
    model_name="personality_model_simple",
    model_version=3,
)
