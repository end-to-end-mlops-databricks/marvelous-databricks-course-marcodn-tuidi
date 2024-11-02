from typing import Any, Dict

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from personality_types.config import ProjectConfig
from personality_types.personality_model import PersonalityModel
from personality_types.utils.delta_utils import get_table_version
from personality_types.utils.logger_utils import set_logger
from personality_types.utils.predictions_utils import custom_predictions

logger = set_logger()


class PersonalityModelProb(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow model wrapper for personality prediction that returns
    both the predicted class labels and their associated probabilities.

    This class wraps a `PersonalityModel` and implements the `predict` method
    required by MLflows, allowing it to be used in an MLflow
    model registry and deployed for inference.

    Attributes:
        model (PersonalityModel): The personality model that performs
            predictions and returns class probabilities.
        config (ProjectConfig): A configuration object containing model's
            hyperparameters.
    """

    def __init__(self, model: PersonalityModel, config: ProjectConfig):
        """
        Initializes the PersonalityModelProb with a pre-trained
        PersonalityModel.

        Args:
            model (PersonalityModel): An instance of the PersonalityModel
                class that provides `predict` and `predict_proba` methods for
                inference.
            config (ProjectConfig): A configuration object containing model's
                hyperparameters.
        """
        self.model = model
        self.config = config

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> Any:
        """
        Predicts personality class labels and their associated probabilities.

        Args:
            context (PythonModelContext): MLflow context, providing access to
                model artifacts and environment configuration.
            model_input (pd.DataFrame): Input features for the model, expected
                as a pandas DataFrame.

        Returns:
            Any: A dictionary containing predicted class labels and their
            maximum probabilities, structured as:
                - "class": Predicted class labels (array-like).
                - "prob": Maximum probability for each prediction (array-like).

        Raises:
            ValueError: If `model_input` is not a pandas DataFrame.
        """
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            prob_prediction = self.model.predict_proba(model_input)
            return custom_predictions(predictions, prob_prediction)
        else:
            raise ValueError("Input must be a pandas DataFrame.")

    def log_model(
        self,
        spark: SparkSession,
        experiment_name: str,
        run_tags: Dict[str, Any],
        model_version_alias: str,
    ) -> ModelVersion:
        """
        Logs the personality prediction model to MLflow, including metadata,
        artifacts, and environment configuration, and registers it in the
        MLflow Model Registry.

        Args:
            spark (SparkSession): The active Spark session for loading data.
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
        client = MlflowClient()

        shema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_table_path = f"{shema_path}.train_set"

        drop_columns_train = ["id", "update_timestamp_utc", self.config.target]

        logger.info(f"Load train data from {train_table_path}")
        train_set_spark = spark.table(train_table_path)

        X_train = train_set_spark.drop(*drop_columns_train).toPandas()

        model_name = f"{shema_path}.personality_model_prob"

        whl_name = "mlops_with_databricks-0.0.1-py3-none-any.whl"
        whl_path = f"code/{whl_name}"
        code_path = "dist/" + whl_name

        example_input = X_train.iloc[0:1]
        example_prediction = self.predict(
            context=None, model_input=example_input
        )

        logger.info("Configuring mlflow to log on databricks")
        mlflow.set_tracking_uri("databricks://adb-tuidiworkspace")
        mlflow.set_registry_uri("databricks-uc://adb-tuidiworkspace")

        logger.info(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name=experiment_name)

        logger.info("Start run")
        with mlflow.start_run(tags=run_tags) as run:
            run_id = run.info.run_id

            logger.info("Log model input.")
            signature = infer_signature(
                model_input=X_train, model_output=example_prediction
            )
            table_version = get_table_version(spark, train_table_path)
            dataset = mlflow.data.from_spark(
                train_set_spark,
                table_name=train_table_path,
                version=table_version,
            )
            mlflow.log_input(dataset, context="training")

            logger.info("Define conda environment.")
            _mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=[whl_path],
                additional_conda_channels=None,
            )

            logger.info("Log model.")
            mlflow.pyfunc.log_model(
                python_model=self,
                artifact_path="randomforest-pipeline-model-prob",
                code_path=[code_path],
                signature=signature,
            )

        logger.info("Register model.")
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/randomforest-pipeline-model-prob",
            name=model_name,
            tags=run_tags,
        )

        logger.info("Set model alias.")
        client.set_registered_model_alias(
            model_name, model_version_alias, f"{model_version.version}"
        )

        return model_version
