# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/personality_types/package/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from personality_types.config import ProjectConfig
from personality_types.registered_model_serving import create_model_serving

# COMMAND ----------
workspace = WorkspaceClient()
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
schema_path = f"{config.catalog_name}.{config.schema_name}"

online_table_name = f"{schema_path}.personality_features_online"
spec = OnlineTableSpec(
    primary_key_columns=["id"],
    source_table_full_name=f"{schema_path}.personality_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
        {"triggered": "true"}
    ),
    perform_full_copy=False,
)

online_table_pipeline = workspace.online_tables.create(
    name=online_table_name,
    spec=spec
)

# COMMAND ----------
create_model_serving(
    config=config,
    workspace=workspace,
    endpoint_name="personality-types-fe-model-serving",
    model_name="personality_model_fe",
    model_version=4,
)
