# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig, Tags
from house_price.models.custom_model import CustomModel

from house_price import __version__ as house_price_v

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
# Initialize model with the config path
custom_model = CustomModel(
    config=config, tags=tags, spark=spark,
    code_paths=[f"../dist/house_price-{house_price_v}-py3-none-any.whl"]
)

# COMMAND ----------
custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
custom_model.train()
custom_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(experiment_names=["/Shared/house-prices-custom"]).run_id[0]

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-house-price-model")

# COMMAND ----------
# Retrieve dataset for the current run
custom_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
custom_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
custom_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = custom_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
