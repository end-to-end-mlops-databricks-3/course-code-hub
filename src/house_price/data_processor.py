"""Data preprocessing module."""

import datetime
import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from house_price.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        # Handle missing values and convert data types as needed
        self.df["LotFrontage"] = pd.to_numeric(self.df["LotFrontage"], errors="coerce")

        self.df["GarageYrBlt"] = pd.to_numeric(self.df["GarageYrBlt"], errors="coerce")
        median_year = self.df["GarageYrBlt"].median()
        self.df["GarageYrBlt"].fillna(median_year, inplace=True)
        current_year = datetime.datetime.now().year

        self.df["GarageAge"] = current_year - self.df["GarageYrBlt"]
        self.df.drop(columns=["GarageYrBlt"], inplace=True)

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Fill missing values with mean or default values
        self.df.fillna(
            {
                "LotFrontage": self.df["LotFrontage"].mean(),
                "MasVnrType": "None",
                "MasVnrArea": 0,
            },
            inplace=True,
        )

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = cat_features + num_features + [target] + ["Id"]
        self.df = self.df[relevant_columns]
        self.df["Id"] = self.df["Id"].astype("str")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic data matching input DataFrame distributions with optional drift.

    Creates artificial dataset replicating statistical patterns from source columns including numeric,
    categorical, and datetime types. Supports intentional data drift for specific features when enabled.

    :param df: Source DataFrame containing original data distributions
    :param drift: Flag to activate synthetic data drift injection
    :param num_rows: Number of synthetic records to generate
    :return: DataFrame containing generated synthetic data
    """
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Id":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if column in {"YearBuilt", "YearRemodAdd"}:  # Handle year-based columns separately
                synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)
            else:
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

                if column == "SalePrice":
                    synthetic_data[column] = np.maximum(0, synthetic_data[column])  # Ensure values are non-negative

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to integers
    int_columns = {
        "LotArea",
        "OverallQual",
        "OverallCond",
        "GarageCars",
        "SalePrice",
        "YearBuilt",
        "YearRemodAdd",
        "TotalBsmtSF",
        "GrLivArea",
    }
    for col in int_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.int64)

    # Only process columns if they exist in synthetic_data
    for col in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
        if col in synthetic_data.columns:
            synthetic_data[col] = pd.to_numeric(synthetic_data[col], errors="coerce")
            synthetic_data[col] = synthetic_data[col].astype(np.float64)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["Id"] = [str(timestamp_base + i) for i in range(num_rows)]

    if drift:
        # Skew the top features to introduce drift
        top_features = ["OverallQual", "GrLivArea"]  # Select top 2 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 2

        # Set YearBuilt to within the last 2 years
        current_year = pd.Timestamp.now().year
        if "YearBuilt" in synthetic_data.columns:
            synthetic_data["YearBuilt"] = np.random.randint(current_year - 2, current_year + 1, num_rows)

    return synthetic_data


def generate_test_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 100) -> pd.DataFrame:
    """Generate test data matching input DataFrame distributions with optional drift."""
    return generate_synthetic_data(df, drift, num_rows)
