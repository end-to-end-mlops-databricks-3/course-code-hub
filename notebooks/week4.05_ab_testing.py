# Databricks notebook source
# MAGIC %pip install house_price-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import hashlib

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig, Tags
from house_price.models.basic_model import BasicModel
from marvelous.common import is_databricks
from dotenv import load_dotenv
import os
import requests
import time

# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="prd")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week4"})

# COMMAND ----------
# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# train model A
basic_model = BasicModel(config=config, tags=tags, spark=spark)
basic_model.load_data()
basic_model.prepare_features()
basic_model.train()
basic_model.log_model()
basic_model.register_model()
model_A_uri = f"models:/{basic_model.model_name}@latest-model"

# COMMAND ----------
# train model B
basic_model_b = BasicModel(config=config, tags=tags, spark=spark)
basic_model_b.paramaters = {"learning_rate": 0.01,
                            "n_estimators": 1000,
                            "max_depth": 6}
basic_model_b.model_name = f"{catalog_name}.{schema_name}.house_prices_model_basic_B"
basic_model_b.load_data()
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
basic_model_b.register_model()
model_B_uri = f"models:/{basic_model_b.model_name}@latest-model"

# COMMAND ----------
# define wrapper
class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model_a = mlflow.sklearn.load_model(
            context.artifacts["lightgbm-pipeline-model-A"]
        )
        self.model_b = mlflow.sklearn.load_model(
            context.artifacts["lightgbm-pipeline-model-B"]
        )

    def predict(self, context, model_input):
        house_id = str(model_input["Id"].values[0])
        hashed_id = hashlib.md5(house_id.encode(encoding="UTF-8")).hexdigest()
        # convert a hexadecimal (base-16) string into an integer
        if int(hashed_id, 16) % 2:
            predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model A"}
        else:
            predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model B"}

# COMMAND ----------
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
X_train = train_set[config.num_features + config.cat_features + ["Id"]]
X_test = test_set[config.num_features + config.cat_features + ["Id"]]

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/house-prices-ab-testing")
model_name = f"{catalog_name}.{schema_name}.house_prices_model_pyfunc_ab_test"
wrapped_model = HousePriceModelWrapper()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 1234.5, "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-house-price-model-ab",
        artifacts={
            "lightgbm-pipeline-model-A": model_A_uri,
            "lightgbm-pipeline-model-B": model_B_uri},
        signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-house-price-model-ab", name=model_name, tags=tags.dict()
)

# COMMAND ----------
"""Model serving module."""

workspace = WorkspaceClient()
model_name=f"{catalog_name}.{schema_name}.house_prices_model_custom_db"
endpoint_name="house-prices-custom-model-serving-db"
entity_version = model_version.version # registered model version

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled="True",
        workload_size="Small",
        entity_version=entity_version,
    )
]

workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
    )

# COMMAND ----------

# Create a sample request body
required_columns = [
    "LotFrontage",
    "LotArea",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "TotalBsmtSF",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Neighborhood",
    "Condition1",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "SaleType",
    "SaleCondition",
    "Id",
]

spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

print(train_set.dtypes)
print(dataframe_records[0])

# COMMAND ----------

# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'LotFrontage': 78.0,
  'LotArea': 9317,
  'OverallQual': 6,
  'OverallCond': 5,
  'YearBuilt': 2006,
  'Exterior1st': 'VinylSd',
  'Exterior2nd': 'VinylSd',
  'MasVnrType': 'None',
  'Foundation': 'PConc',
  'Heating': 'GasA',
  'CentralAir': 'Y',
  'SaleType': 'WD',
  'SaleCondition': 'Normal'}]
"""

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/house-prices-custom-model-serving-db/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)