from typing import Any, Dict

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer

from personality_types.config import ProjectConfig
from personality_types.personality_model import PersonalityModel
from personality_types.utils.logger_utils import set_logger

logger = set_logger()


def fe_logger(
    spark: SparkSession,
    config: ProjectConfig,
    fe: FeatureEngineeringClient,
    preprocessor: ColumnTransformer,
    experiment_name: str,
    run_tags: Dict[str, Any],
    training_set: Any,
) -> ModelVersion:
    """
    Trains a personality classification model using a feature engineering
    client for data loading, evaluates it, and logs parameters, metrics, and
    artifacts to MLflow, registering the model in the MLflow Model Registry.

    Args:
        spark (SparkSession): The active Spark session for data loading.
        config (ProjectConfig): Configuration object containing model
            hyperparameters and settings.
        fe (FeatureEngineeringClient): Feature engineering client to access
            the training set and log the model.
        preprocessor (ColumnTransformer): Preprocessing steps to transform
            input features before training.
        experiment_name (str): The name of the MLflow experiment for logging.
        run_tags (Dict[str, Any]): Metadata tags to associate with the
            MLflow run.
        training_set (Any): Training set loaded through the feature engineering
            client for model training.

    Returns:
        ModelVersion: The versioned model registered in the MLflow Model
            Registry.
    """
    training_df = training_set.load_df().toPandas()
    shema_path = f"{config.catalog_name}.{config.schema_name}"
    test_table_path = f"{shema_path}.test_set"

    drop_columns_train = ["id", "update_timestamp_utc", config.target]

    logger.info(f"Load test data from {test_table_path}")
    test_set_spark = spark.table(test_table_path)

    X_train = training_df.drop(["id", config.target], axis=1)
    X_test = test_set_spark.drop(*drop_columns_train).toPandas()
    X_test["score_avg"] = X_test[["thinking_score", "sensing_score"]].mean(
        axis=1
    )

    y_train = training_df[config.target]
    y_test = test_set_spark.select(config.target).toPandas()

    logger.info("Model definition")
    model = PersonalityModel(preprocessor, config)

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
        model.train(X_train, y_train)

        logger.info("Get predictions")
        y_pred = model.predict(X_test)

        accuracy = model.evaluate(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy}")

        mlflow.log_param(
            "model_type", "Random forest classifier with preprocessing"
        )

        mlflow.log_params(config.parameters)

        mlflow.log_metric("accuracy", accuracy)

        feature_importance_plot = model.get_feature_importance_plot()
        mlflow.log_figure(
            feature_importance_plot, "plots/feature_importance.png"
        )

        signature = infer_signature(model_input=X_train, model_output=y_pred)

        fe.log_model(
            model=model,
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
