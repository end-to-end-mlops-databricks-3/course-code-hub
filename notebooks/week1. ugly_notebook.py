# Databricks notebook source
# MAGIC %md
# MAGIC # House Price Prediction Exercise
# MAGIC
# MAGIC This notebook demonstrates how to predict house prices using the house price dataset. We'll go through the process of loading data, preprocessing, model creation, and visualization of results.
# MAGIC
# MAGIC ## Importing Required Libraries
# MAGIC
# MAGIC First, let's import all the necessary libraries.

# COMMAND ----------

import datetime

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

# COMMAND ----------

# Load configuration
with open("../project_config.yml") as file:
    config = yaml.safe_load(file)

catalog_name = config["catalog_name"]
schema_name = config["schema_name"]


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Only works in a Databricks environment if the data is there
# to put data there, create volume and run databricks fs cp <path> dbfs:/Volumes/mlops_dev/<schema_name>/<volume_name>/
# filepath = f"/Volumes/{catalog_name}/{schema_name}/data/data.csv"
# Load the data
# df = pd.read_csv(filepath)

# Works both locally and in a Databricks environment
filepath = "../data/data.csv"
# Load the data
df = pd.read_csv(filepath)

# Works both locally and in a Databricks environment
# df = spark.read.csv(f"/Volumes/{catalog_name}/{schema_name}/data/data.csv", header=True, inferSchema=True).toPandas()


# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

# Remove rows with missing target

# Handle missing values and convert data types as needed
df["LotFrontage"] = pd.to_numeric(df["LotFrontage"], errors="coerce")

df["GarageYrBlt"] = pd.to_numeric(df["GarageYrBlt"], errors="coerce")
median_year = df["GarageYrBlt"].median()
df["GarageYrBlt"].fillna(median_year, inplace=True)
current_year = datetime.datetime.now().year

df["GarageAge"] = current_year - df["GarageYrBlt"]
df.drop(columns=["GarageYrBlt"], inplace=True)

# Handle numeric features
num_features = config["num_features"]
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values with mean or default values
df.fillna(
    {
        "LotFrontage": df["LotFrontage"].mean(),
        "MasVnrType": "None",
        "MasVnrArea": 0,
    },
    inplace=True,
)

# Convert categorical features to the appropriate type
cat_features = config["cat_features"]
for cat_col in cat_features:
    df[cat_col] = df[cat_col].astype("category")

# Extract target and relevant features
target = config["target"]
relevant_columns = cat_features + num_features + [target] + ["Id"]
df = df[relevant_columns]
df["Id"] = df["Id"].astype("str")

# COMMAND ----------

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

train_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")

test_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.train_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.test_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# COMMAND ----------
