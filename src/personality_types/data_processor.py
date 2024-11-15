from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from personality_types.config import ProjectConfig
from personality_types.custom_transforms import (
    EducationTransform,
    GenderTransform,
)


class DataProcessor:
    """
    A class responsible for loading, processing, and splitting data
    for machine learning models.

    Attributes:
        df (pd.DataFrame): Loaded dataset as a pandas DataFrame.
        train (bool): A flag indicating whether the data is for training
            (True) or inference (False).
        config (ProjectConfig): A configuration object containing feature
            names and other processing parameters.
        X (Optional[pd.DataFrame]): The feature matrix.
        y (Optional[pd.Series]): The target vector.
        preprocessor (Optional[ColumnTransformer]): The preprocessor used for
            data transformation.
    """

    def __init__(
        self,
        spark: SparkSession,
        data_path: str,
        train: bool,
        config: ProjectConfig,
    ) -> None:
        """
        Initializes the DataProcessor with the data path, training flag,
        and configuration.

        Args:
            spark (SparkSession): current spark session.
            data_path (str): Path to the CSV file containing the dataset.
            train (bool): A flag indicating whether it's training mode (True)
                or inference mode (False).
            config (dict): Configuration dictionary containing feature names
                and target variable details.
        """
        self.df = self.rename_columns(self.load_data(spark, data_path))
        self.train = train
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, spark: SparkSession, data_path: str) -> pd.DataFrame:
        """
        Loads the dataset from the specified CSV file path.

        Args:
            spark (SparkSession): current spark session.
            data_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        return spark.read.csv(data_path, header=True).toPandas()

    def load_delta(self, spark: SparkSession, table_name: str) -> DataFrame:
        """
        Loads the dataset from the specified delta table in catalog.

        Args:
            spark (SparkSession): current spark session.
            table_name (str): Name of the table to load.

        Returns:
            DataFrame: Loaded dataset.
        """
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        return spark.table(f"{schema_path}.{table_name}")

    @staticmethod
    def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames columns of a pandas dataframe so that columns names are
        lowercase and follow snake case convention.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df

    def create_target(self, target: str, raw_target: str) -> None:
        """
        Creates the target column in the dataframe by mapping raw target types
        to groups.

        Args:
            target (str): The name of the new target column.
            raw_target (str): The column containing the raw personality types.
        """
        anlaysts = ["INTJ", "INTP", "ENTJ", "ENTP"]
        diplomats = ["INFJ", "INFP", "ENFJ", "ENFP"]
        sentinels = ["ISTJ", "ISFJ", "ESTJ", "ESFJ"]
        explorers = ["ISTP", "ISFP", "ESTP", "ESFP"]

        self.df[target] = np.where(
            self.df[raw_target].isin(anlaysts),
            "Analyst",
            np.where(
                self.df[raw_target].isin(diplomats),
                "Diplomat",
                np.where(
                    self.df[raw_target].isin(sentinels),
                    "Sentinel",
                    np.where(
                        self.df[raw_target].isin(explorers), "Explorer", None
                    ),
                ),
            ),
        )

    def add_id_column(self):
        """
        Add string id columns to dataframe.
        """
        self.X["id"] = np.arange(len(self.X))
        self.X["id"] = self.X["id"].astype(str)

    def preprocess_data(self) -> None:
        """
        Preprocesses the dataset by handling missing values, scaling numeric
        features, and encoding categorical features.
        The preprocessor is built and stored as an attribute, and `X` and `y`
        are set up based on the feature matrix and target.
        """
        num_features = self.config.num_features
        cat_features = self.config.cat_features
        train_features = num_features + cat_features

        # remove columns with missing raw target
        if self.train:
            raw_target = self.config.raw_target
            target = self.config.target

            self.create_target(target, raw_target)
            self.df = self.df.dropna(subset=[target])

            self.y = self.df[target]

        self.X = self.df[train_features]

        # train features preprocessing steps
        # numeric features
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # categorical features
        standard_categorical = list(
            set(self.config.cat_features) - set(["gender", "education"])
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Unknown"),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        gender_transformer = Pipeline(
            steps=[
                ("force_value", GenderTransform()),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        education_transform = Pipeline(
            steps=[("force_value", EducationTransform())]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_features),
                ("gender", gender_transformer, "gender"),
                ("education", education_transform, "education"),
                ("cat", categorical_transformer, standard_categorical),
            ],
            remainder="passthrough",
        )

    def split_data(
        self,
        test_size: Optional[float] = 0.2,
        random_state: Optional[int] = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and testing sets based on the
        specified test size and random state.

        Args:
            test_size (float): Proportion of the dataset to include in the test
                split. Default is 0.2.
            random_state (int): Random seed for reproducibility. Default is 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The split
                data as (X_train, X_test, y_train, y_test).
        """
        return train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

    def save_to_catalog(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        spark: SparkSession,
    ) -> None:
        """
        Save the train and test sets into unity catalog. Adds update timestamp
        to simulate delta changes.

        Args:
            X_train (pd.DataFrame): Training set to be saved.
            X_test (pd.DataFrame): Test set to be saved.
            y_train (pd.Series): Train target to be saved inside train set.
            y_test (pd.Series): Test target to be saved inside test set.
            spark (SparkSession): current spark session.
        """

        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc",
            F.to_utc_timestamp(F.current_timestamp(), "UTC"),
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc",
            F.to_utc_timestamp(F.current_timestamp(), "UTC"),
        )

        shema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_table_path = f"{shema_path}.train_set"
        test_table_path = f"{shema_path}.test_set"

        train_set_with_timestamp.write.mode("append").saveAsTable(
            train_table_path
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            test_table_path
        )

        spark.sql(
            f"""
            ALTER TABLE {train_table_path}
            SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
            """
        )

        spark.sql(
            f"""
            ALTER TABLE {test_table_path}
            SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
            """
        )

    def create_source_table(
        self, spark: SparkSession, train_table_name: str, test_table_name: str
    ) -> None:
        """
        Save the source table to be used for the rest of the project.
        Source table is the concatenation of train and test data extracted
        from the csv file loaded in the volume.

        Args:
            spark (SparkSession): current spark session.
            train_table_name (str): Name of the train table to load.
            test_table_name (str): Name of the test table to load.

        """
        schema_path = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_table = self.load_delta(spark, train_table_name)
        test_table = self.load_delta(spark, test_table_name)
        source_table = train_table.unionByName(test_table)

        source_table_path = f"{schema_path}.source_table"

        source_table.write.mode("append").saveAsTable(source_table_path)

        spark.sql(
            f"""
            ALTER TABLE {source_table_path}
            SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
            """
        )
