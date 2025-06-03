import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.serving.fe_model_serving import FeatureLookupServing
from databricks.sdk import WorkspaceClient
from marvelous.common import create_parser

args = create_parser()

root_path = args.root_path
is_test = args.is_test
config_path = f"{root_path}/files/project_config.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"house-prices-model-serving-fe-{args.env}"

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.house_prices_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.house_features",
)

# Create or update the online table for house features
feature_model_server.create_or_update_online_table()
logger.info("Created or updated online table.")

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Started deployment/update of the serving endpoint.")

# Delete endpoint if test
if is_test:
    workspace = WorkspaceClient()
    workspace.serving_endpoints.delete(name=endpoint_name)
    logger.info("Deleting serving endpoint.")
