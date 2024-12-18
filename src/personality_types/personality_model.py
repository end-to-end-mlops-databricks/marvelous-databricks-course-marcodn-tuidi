from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureLookup,
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from matplotlib.figure import Figure
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from personality_types.config import ProjectConfig
from personality_types.utils.delta_utils import get_table_version
from personality_types.utils.logger_utils import set_logger

logger = set_logger()


class PersonalityModel(mlflow.pyfunc.PythonModel):
    """
    A class that constructs a machine learning pipeline to preprocess data,
    train a random forest classifier, and make predictions for personality
    type classification.

    Attributes:
        config (ProjectConfig): A configuration object containing model's
            hyperparameters.
        model (Pipeline): Pipeline containing a preprocessing step and a
        random forest classifier.
    """

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        config: ProjectConfig,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the PersonalityModel with a specified preprocessor and
        model configuration.

        Args:
            preprocessor (ColumnTransformer): Preprocessing steps to transform
                input features before training.
            config (ProjectConfig): A configuration object containing model's
                hyperparameters.
            model_path (Optional, str): Path of the model to load. If not None
                the model will be loaded and not created.
        """
        self.config = config
        base_model = RandomForestClassifier(
            n_estimators=config.parameters["n_estimators"],
            max_depth=config.parameters["max_depth"],
            random_state=42,
        )
        if model_path is not None:
            model = mlflow.pyfunc.load_model(model_path)
            self.model = model.unwrap_python_model()
        else:
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", base_model),
                ]
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Method that trains the RandomForest model on the provided training
        data.

        Args:
            X_train (pd.DataFrame): Training dataset containing the training
                features.
            y_train (pd.Series): Training target.
        """
        self.model.fit(X_train, y_train)

    def predict(
        self, context: PythonModelContext, model_input: Any
    ) -> np.ndarray:
        """
        Method that generates predictions using the trained model on the
        provided data.

        Args:
            model_input (Any): Data to use for making the predictions.
                This can be in various formats such as dictionary, list, or
                DataFrame as received from the endpoint.

        Returns:
            np.ndarray: Array of predicted classes.
        """
        # Handle different input formats
        if isinstance(model_input, dict):
            # If using "dataframe_split" format
            if "data" in model_input and "columns" in model_input:
                model_input = pd.DataFrame(
                    data=model_input["data"], columns=model_input["columns"]
                )

            # If using "dataframe_records" format
            elif "dataframe_records" in model_input:
                model_input = pd.DataFrame(model_input["dataframe_records"])

        elif isinstance(model_input, list):
            # If a list of lists, convert it to DataFrame with default columns
            # if needed
            model_input = pd.DataFrame(model_input)

        elif not isinstance(model_input, pd.DataFrame):
            # If not in expected format, raise an error
            raise ValueError("Unsupported input format for model prediction")
        """if isinstance(self, PersonalityModel):
            model = self.model
        else:
            model = self.unwrap_python_model()"""
        predictions = self.model.predict(model_input)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Method that generates probability predictions using the trained model
        on the provided data.

        Args:
            X (pd.DataFrame): Data to use for making the predictions.

        Returns:
            np.ndarray: Array of predicted classes.
        """
        return self.model.predict_proba(X)

    def evaluate(self, y_test: pd.Series, y_pred: pd.Series) -> float:
        """
        Evaluates the model's accuracy based on provided test labels and
        predictions.

        Args:
            y_test (pd.Series): True labels for the test data.
            y_pred (pd.Series): Predicted labels from the model.

        Returns:
            float: The accuracy score of the model.
        """
        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        return accuracy

    def get_feature_importance_plot(self) -> Figure:
        """
        Generates a bar plot of feature importances from the Random Forest
        model.

        Returns:
            Figure: A matplotlib figure containing the feature importance plot.
        """
        preprocessor = self.model.named_steps["preprocessor"]
        model = self.model.named_steps["classifier"]
        importances = model.feature_importances_
        features = preprocessor.get_feature_names_out()
        feature_importance_df = pd.DataFrame(
            {"Feature": features, "Importance": importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
        plt.title("Feature Importances from Random Forest Classifier")
        return fig

    def train_and_log(
        self,
        spark: SparkSession,
        experiment_name: str,
        run_tags: Dict[str, Any],
        model_name: str,
        model_alias: Optional[str] = None,
    ) -> ModelVersion:
        """
        Trains the model, evaluates it, and logs parameters, metrics, and
        artifacts to MLflow, including model registry.

        Args:
            spark (SparkSession): The active Spark session for loading data.
            experiment_name (str): The name of the MLflow experiment.
            run_tags (Dict[str, Any]): Metadata tags for the MLflow run.
            model_name (str): Name of the registered model.
            model_alias (Optional, str): Model alias. If None, no alias is
                assigned to the model. Default None.

        Returns:
            ModelVersion: The versioned model registered in MLflow.
        """
        shema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_table_path = f"{shema_path}.train_set"
        test_table_path = f"{shema_path}.test_set"

        drop_columns_train = ["id", "update_timestamp_utc", self.config.target]

        logger.info(f"Load train data from {train_table_path}")
        train_set_spark = spark.table(train_table_path)

        logger.info(f"Load test data from {test_table_path}")
        test_set_spark = spark.table(test_table_path)

        X_train = train_set_spark.drop(*drop_columns_train).toPandas()
        X_test = test_set_spark.drop(*drop_columns_train).toPandas()

        y_train = train_set_spark.select(self.config.target).toPandas()
        y_test = test_set_spark.select(self.config.target).toPandas()

        logger.info(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name=experiment_name)

        logger.info("Start run")
        with mlflow.start_run(tags=run_tags) as run:
            run_id = run.info.run_id
            logger.info(f"Start run (id: {run_id})")

            logger.info("Training the model...")
            self.train(X_train, y_train)

            logger.info("Get predictions")
            y_pred = self.predict(None, X_test)

            accuracy = self.evaluate(y_test, y_pred)
            logger.info(f"Accuracy: {accuracy}")

            mlflow.log_param(
                "model_type", "Random forest classifier with preprocessing"
            )

            mlflow.log_params(self.config.parameters)

            mlflow.log_metric("accuracy", accuracy)

            feature_importance_plot = self.get_feature_importance_plot()
            mlflow.log_figure(
                feature_importance_plot, "plots/feature_importance.png"
            )

            signature = infer_signature(
                model_input=X_train, model_output=y_pred
            )

            table_version = get_table_version(spark, train_table_path)
            dataset = mlflow.data.from_spark(
                train_set_spark,
                table_name=train_table_path,
                version=table_version,
            )
            mlflow.log_input(dataset, context="training")

            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="randomforest-pipeline-model-simple",
                signature=signature,
            )

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/randomforest-pipeline-model-simple",
            name=f"{shema_path}.{model_name}",
            tags=run_tags,
        )

        if model_alias is not None:
            mlflow.MlflowClient().set_registered_model_alias(
                f"{shema_path}.{model_name}",
                model_alias,
                f"{model_version.version}",
            )

        return model_version

    def train_and_log_custom(
        self,
        spark: SparkSession,
        experiment_name: str,
        run_tags: Dict[str, Any],
        model_name: str,
    ) -> ModelVersion:
        """
        Trains the model, evaluates it, and logs parameters, metrics, and
        artifacts to MLflow, including model registry. Supports custom defined
        model (example: custom preropcessing).

        Args:
            spark (SparkSession): The active Spark session for loading data.
            experiment_name (str): The name of the MLflow experiment.
            run_tags (Dict[str, Any]): Metadata tags for the MLflow run.
            model_name (str): Name of the registered model.

        Returns:
            ModelVersion: The versioned model registered in MLflow.
        """
        shema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_table_path = f"{shema_path}.train_set"
        test_table_path = f"{shema_path}.test_set"

        drop_columns_train = ["id", "update_timestamp_utc", self.config.target]

        logger.info(f"Load train data from {train_table_path}")
        train_set_spark = spark.table(train_table_path)

        logger.info(f"Load test data from {test_table_path}")
        test_set_spark = spark.table(test_table_path)

        X_train = train_set_spark.drop(*drop_columns_train).toPandas()
        X_test = test_set_spark.drop(*drop_columns_train).toPandas()

        y_train = train_set_spark.select(self.config.target).toPandas()
        y_test = test_set_spark.select(self.config.target).toPandas()

        whl_name = "mlops_with_databricks-0.0.1-py3-none-any.whl"
        whl_path = f"code/{whl_name}"
        code_path = "dist/" + whl_name

        logger.info(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name=experiment_name)

        logger.info("Start run")
        with mlflow.start_run(tags=run_tags) as run:
            run_id = run.info.run_id
            logger.info(f"Start run (id: {run_id})")

            logger.info("Training the model...")
            self.train(X_train, y_train)

            logger.info("Get predictions")
            y_pred = self.predict(None, X_test)

            accuracy = self.evaluate(y_test, y_pred)
            logger.info(f"Accuracy: {accuracy}")

            mlflow.log_param(
                "model_type", "Random forest classifier with preprocessing"
            )

            mlflow.log_params(self.config.parameters)

            mlflow.log_metric("accuracy", accuracy)

            feature_importance_plot = self.get_feature_importance_plot()
            mlflow.log_figure(
                feature_importance_plot, "plots/feature_importance.png"
            )

            signature = infer_signature(
                model_input=X_train, model_output=y_pred
            )

            table_version = get_table_version(spark, train_table_path)
            dataset = mlflow.data.from_spark(
                train_set_spark,
                table_name=train_table_path,
                version=table_version,
            )
            mlflow.log_input(dataset, context="training")

            logger.info("Define conda environment.")
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
                artifact_path="randomforest-pipeline-model",
                code_path=[code_path],
                signature=signature,
                conda_env=conda_env,
            )

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/randomforest-pipeline-model",
            name=f"{shema_path}.{model_name}",
            tags=run_tags,
        )

        return model_version

    def train_and_log_from_fe(
        self,
        spark: SparkSession,
        fe: FeatureEngineeringClient,
        experiment_name: str,
        run_tags: Dict[str, Any],
        training_set: Any,
    ) -> ModelVersion:
        """
        Trains the model using data from the Feature Engineering client,
        evaluates, and logs artifacts and metrics to MLflow.

        Args:
            spark (SparkSession): The active Spark session for loading data.
            fe (FeatureEngineeringClient): Client for feature engineering and
                feature store access.
            experiment_name (str): The name of the MLflow experiment.
            run_tags (Dict[str, Any]): Metadata tags for the MLflow run.
            training_set (Any): Dataset provided by FeatureEngineeringClient
                for training.

        Returns:
            ModelVersion: The versioned model registered in MLflow.
        """
        training_df = training_set.load_df().toPandas()
        shema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        test_table_path = f"{shema_path}.test_set"

        drop_columns_train = ["id", "update_timestamp_utc", self.config.target]

        logger.info(f"Load test data from {test_table_path}")
        test_set_spark = spark.table(test_table_path)

        X_train = training_df.drop(["id", self.config.target], axis=1)
        X_test = test_set_spark.drop(*drop_columns_train).toPandas()
        X_test["score_avg"] = X_test[["thinking_score", "sensing_score"]].mean(
            axis=1
        )

        y_train = training_df[self.config.target]
        y_test = test_set_spark.select(self.config.target).toPandas()

        logger.info(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name=experiment_name)

        logger.info("Start run")
        with mlflow.start_run(tags=run_tags) as run:
            run_id = run.info.run_id
            logger.info(f"Start run (id: {run_id})")

            logger.info("Training the model...")
            self.train(X_train, y_train)

            logger.info("Get predictions")
            y_pred = self.predict(None, X_test)

            accuracy = self.evaluate(y_test, y_pred)
            logger.info(f"Accuracy: {accuracy}")

            mlflow.log_param(
                "model_type", "Random forest classifier with preprocessing"
            )

            mlflow.log_params(self.config.parameters)

            mlflow.log_metric("accuracy", accuracy)

            feature_importance_plot = self.get_feature_importance_plot()
            mlflow.log_figure(
                feature_importance_plot, "plots/feature_importance.png"
            )

            signature = infer_signature(
                model_input=X_train, model_output=y_pred
            )

            fe.log_model(
                model=self.model,
                flavor=mlflow.sklearn,
                artifact_path="randomforest-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/randomforest-pipeline-model-fe",
            name=f"{shema_path}.personality_model_fe",
            tags=run_tags,
        )

        return model_version

    def batch_predictions(
        self,
        spark: SparkSession,
        df: pd.DataFrame,
    ) -> DataFrame:
        """
        Creates a feature table containing the pre-computed predictions.

        Args:
            spark (SparkSession): The active Spark session for loading data.
            df (pd.DataFrame): Dataframe to use for prediction.

        Returns:
            DataFrame: Spark DataFrame with the predictions.
        """
        drop_columns = ["id", "update_timestamp_utc", self.config.target]
        df_prediction = df[["id", "gender", "age"]]
        df = df.drop(drop_columns, axis=1)
        df_prediction["predicted_class"] = self.predict(None, df)

        df_prediction_spark = spark.createDataFrame(df_prediction)

        return df_prediction_spark

    def create_bach_feature_serving(
        self,
        spark: SparkSession,
        workspace: WorkspaceClient,
        fe: FeatureEngineeringClient,
        df_prediction: DataFrame,
        feature_table_name: str,
        online_table_name: str,
        feature_spec_name: str,
        endpoint_name: str,
    ):
        """
        Method that creates a serving point fot accessing predictions.

        Args:
            spark (SparkSession): The active Spark session for loading data.
            workspace (WorkspaceClient): Databricks workspace client.
            fe (FeatureEngineeringClient): Client for feature engineering and
                feature store access.
            df_prediction (DataFrame): Spark dataframe with the predictions.
            feature_table_name (str): Name of the feature table.
            online_table_name (str): Name of the online table spec.
            feature_spec_name (str): Name of the feature spec.
            endpoint_name (str): Name of the serving endpoint.
        """
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        feature_table_path = f"{schema_path}.{feature_table_name}"
        online_table_path = f"{schema_path}.{online_table_name}"
        feature_spec_path = f"{schema_path}.{feature_spec_name}"

        # Create feature table containing the predictions
        fe.create_table(
            name=feature_table_path,
            primary_keys=["id"],
            df=df_prediction,
            description="Personality types predictions feature table",
        )
        spark.sql(
            f"""
            ALTER TABLE {feature_table_path}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """
        )

        # Create the online table for fast retrival
        trigger = OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
            {"triggered": "true"}
        )
        spec = OnlineTableSpec(
            primary_key_columns=["id"],
            source_table_full_name=feature_table_path,
            run_triggered=trigger,
            perform_full_copy=False,
        )
        workspace.online_tables.create(name=online_table_path, spec=spec)

        # Create feature lookup
        features = [
            FeatureLookup(
                table_name=feature_table_path,
                lookup_key="id",
                feature_names=["gender", "age", "predicted_class"],
            )
        ]

        # Create feature spec table for serving
        fe.create_feature_spec(
            name=feature_spec_path, features=features, exclude_columns=None
        )

        # Create serving endpoint
        workspace.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[
                    ServedEntityInput(
                        entity_name=feature_spec_path,
                        scale_to_zero_enabled=True,
                        workload_size="Small",
                    )
                ]
            ),
        )
        logger.info("Serving endpoint created.")
