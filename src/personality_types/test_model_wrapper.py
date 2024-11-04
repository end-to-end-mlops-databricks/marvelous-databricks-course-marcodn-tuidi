import hashlib
from typing import Any, Dict

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import DataFrame, SparkSession

from personality_types.config import ProjectConfig
from personality_types.utils.delta_utils import get_table_version


class PersonalityTypesModelWrapper(mlflow.pyfunc.PythonModel):
    """
    A wrapper to conduct a / b testing on two models,

    Attributes:
        config (ProjectConfig): A configuration object containing model's
            hyperparameters.
        models (list): List of models to compare.
    """

    def __init__(self, models: list):
        """
        Initializes the model wrapper class.

        Args:
            config (ProjectConfig): A configuration object containing model's
                hyperparameters.
            models (list): List of models to compare.
        """
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
        """
        Model wrapper predictions for a/b testing. The input data is id hashed
        and then converted to int. Even transformed id are predicted by model
        a and odds ones by model b. This ensures traffic split and consistency
        between then input and then model used for prediction.

        Args:
            contect (PythonModelContext): : MLflow context, providing access to
                model artifacts and environment configuration.
            model_input (pd.DataFrame): Input features for the model, expected
                as a pandas DataFrame.

        Returns:
            Any: A dictionary containing predicted class labels and the model
            used for the prediction.
                - "Prediction": Predicted class.
                - "model": Used model.

        Raises:
            ValueError: If `model_input` is not a pandas DataFrame.
        """
        if isinstance(model_input, pd.DataFrame):
            input_id = str(model_input["id"].values[0])
            hashed_id = hashlib.md5(
                input_id.encode(encoding="UTF-8")
            ).hexdigest()
            if int(hashed_id, 16) % 2:
                predictions = self.model_a.predict(
                    model_input.drop(["id"], axis=1)
                )
                return {
                    "Prediction": predictions[0],
                    "model": "Model A",
                }
            else:
                predictions = self.model_b.predict(
                    model_input.drop(["id"], axis=1)
                )
                return {
                    "Prediction": predictions[0],
                    "model": "Model B",
                }
        else:
            raise ValueError("Input must be a pandas DataFrame.")

    def log_model(
        self,
        spark: SparkSession,
        config: ProjectConfig,
        train_set_spark: DataFrame,
        experiment_name: str,
        run_tags: Dict[str, Any],
        model_name: str,
    ) -> int:
        """
        Logs the wrapper model to MLflow, including metadata,
        artifacts, and environment configuration, and registers it in the
        MLflow Model Registry.

        Args:
            spark (SparkSession): The active Spark session for loading data.
            config (ProjectConfig): A configuration object containing model's
                hyperparameters.
            train_set_spark (DataFrame): Example of train set dataframe for
                input data logging and signature inference.
            experiment_name (str): The MLflow experiment name for logging
                model information.
            run_tags (Dict[str, Any]): Tags for the MLflow run, adding metadata
                to the logged model.
            model_version_alias (str): Alias for the registered model version,
                allowing easier access to the model in production.

        Returns:
            ModelVersion: The registered model version in the MLflow Model
            Registry with the specified alias.
        """
        schema_path = f"{config.catalog_name}.{config.schema_name}"
        mlflow.set_experiment(experiment_name=experiment_name)
        model_name = f"{schema_path}.{model_name}"

        drop_columns_train = ["update_timestamp_utc", config.target]
        example_input = train_set_spark.drop(*drop_columns_train).toPandas()

        train_table_path = f"{schema_path}.train_set"

        whl_name = "mlops_with_databricks-0.0.1-py3-none-any.whl"
        whl_path = f"code/{whl_name}"
        code_path = "dist/" + whl_name

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            signature = infer_signature(
                model_input=example_input,
                model_output={
                    "Prediction": 1234.5,
                    "model": "Model B",
                },
            )
            table_version = get_table_version(spark, train_table_path)
            dataset = mlflow.data.from_spark(
                train_set_spark,
                table_name=train_table_path,
                version=table_version,
            )
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=[
                    whl_path,
                    "pyspark==3.5.0",
                    "delta-spark==3.0.0",
                ],
                additional_conda_channels=None,
            )

            mlflow.pyfunc.log_model(
                python_model=self,
                artifact_path="pyfunc-personality-types-model-ab",
                code_path=[code_path],
                signature=signature,
                conda_env=conda_env,
            )

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/pyfunc-personality-types-model-ab",
            name=model_name,
            tags=run_tags,
        )

        return model_version
