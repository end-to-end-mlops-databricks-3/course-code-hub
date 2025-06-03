# Databricks notebook source
# MAGIC %pip install house_price-1.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.serving.feature_serving import FeatureServing

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

fe = feature_engineering.FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name
feature_table_name = f"{catalog_name}.{schema_name}.house_prices_preds"
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"
endpoint_name = "house-prices-feature-serving"

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
df = pd.concat([train_set, test_set])

model = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.house_prices_model_basic@latest-model")


preds_df = df[["Id", "GrLivArea", "YearBuilt"]]
preds_df["Predicted_SalePrice"] = model.predict(df[config.cat_features + config.num_features])
preds_df = spark.createDataFrame(preds_df)

fe.create_table(
    name=feature_table_name, primary_keys=["Id"], df=preds_df, description="House Prices predictions feature table"
)

spark.sql(f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

# Initialize feature store manager
feature_serving = FeatureServing(
    feature_table_name=feature_table_name, feature_spec_name=feature_spec_name, endpoint_name=endpoint_name
)


# COMMAND ----------
# Create online table
feature_serving.create_online_table()

# COMMAND ----------
# Create feature spec
feature_serving.create_feature_spec()

# COMMAND ----------
# Deploy feature serving endpoint
feature_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

start_time = time.time()
serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_records": [{"Id": "182"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------
# another way to call the endpoint

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_split": {"columns": ["Id"], "data": [["182"]]}},
)
