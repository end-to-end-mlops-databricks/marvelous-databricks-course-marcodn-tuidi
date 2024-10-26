from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.personality_types.config import ProjectConfig
from src.utils.delta_utils import get_table_version
from src.utils.logger_utils import set_logger

logger = set_logger()


class PersonalityModel:
    """
    A class that constructs a machine learning pipeline to preprocess data,
    train a random forest classifier, and make predictions for personality
    type classification.

    Attributes:
        config (ProjectConfig): A configuration object containing model's
            hyperparameters
        model (Pipeline): Pipeline containing a preprocessing step and a
        random forest classifier.
    """

    def __init__(
        self, preprocessor: ColumnTransformer, config: ProjectConfig
    ) -> None:
        """
        Initializes the PersonalityModel with a specified preprocessor and
        model configuration.

        Args:
            preprocessor (ColumnTransformer): Preprocessing steps to transform
                input features before training.
            config (ProjectConfig): A configuration object containing model's
                hyperparameters
        """
        self.config = config
        base_model = RandomForestClassifier(
            n_estimators=config.parameters["n_estimators"],
            max_depth=config.parameters["max_depth"],
            random_state=42,
        )
        self.model = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", base_model)]
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Method that generates predictions using the trained model on the
        provided data.

        Args:
            X (pd.DataFrame): Data to use for making the predictions.

        Returns:
            np.ndarray: Array of predicted classes.
        """
        return self.model.predict(X)

    def evaluate(self, y_test: pd.Series, y_pred: pd.Series) -> Tuple[float]:
        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        return accuracy

    def train_and_log(
        self,
        spark: SparkSession,
        experiment_name: str,
        run_tags: Dict[str, Any],
    ) -> None:
        shema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_table_path = f"{shema_path}.train_set"
        test_table_path = f"{shema_path}.test_set"

        logger.info(f"Load train data from {train_table_path}")
        train_set_spark = spark.table(train_table_path)

        logger.info(f"Load test data from {test_table_path}")
        test_set_spark = spark.table(test_table_path)

        X_train = train_set_spark.drop(self.config.target).toPandas()
        X_test = test_set_spark.drop(self.config.target).toPandas()

        y_train = train_set_spark.select(self.config.target).toPandas()
        y_test = test_set_spark.select(self.config.target).toPandas()

        logger.info("Configuring mlflow to log on databricks")
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

        logger.info(f"Setting experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name=experiment_name)

        logger.info("Start run")
        with mlflow.start_run(tags=run_tags) as run:
            run_id = run.info.run_id
            logger.info(f"Start run (id: {run_id})")

            logger.info("Training the model...")
            self.train(X_train, y_train)

            logger.info("Get predictions")
            y_pred = self.predict(X_test)

            accuracy = self.evaluate(y_test, y_pred)
            logger.info(f"Accuracy: {accuracy}")

            mlflow.log_param(
                "model_type", "Random forest classifier with preprocessing"
            )

            mlflow.log_params(self.config.parameters)

            mlflow.log_metric("accuracy", accuracy)

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
                artifact_path="randomforest-pipeline-model",
                signature=signature,
            )
