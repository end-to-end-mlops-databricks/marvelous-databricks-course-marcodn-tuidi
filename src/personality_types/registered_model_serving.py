from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    TrafficConfig,
)

from personality_types.config import ProjectConfig


def create_model_serving(
    config: ProjectConfig,
    workspace: WorkspaceClient,
    endpoint_name: str,
    model_name: str,
    model_version: int,
) -> None:
    """
    Funcion that creates a serving endpoint for a registered model.

    Args:
        workspace (WorkspaceClient): Databricks workspace client.
        endpoint_name (str): Name of the serving endpoint.
        model_name (str): Name of the model to serve.
        model_version (int): Version of the model to serve.
    """
    schema_path = f"{config.catalog_name}.{config.schema_name}"

    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=f"{schema_path}.{model_name}",
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                    entity_version=model_version,
                )
            ],
            traffic_config=TrafficConfig(
                routes=[
                    Route(
                        served_model_name=f"{model_name}-{model_version}",
                        traffic_percentage=100,
                    )
                ]
            ),
        ),
    )
