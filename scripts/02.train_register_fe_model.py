import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig, Tags
from house_price.models.feature_lookup_model import FeatureLookUpModel
from marvelous.comon import create_parser

args = create_parser()

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Create feature table
# fe_model.create_feature_table()

fe_model.update_feature_table()
logger.info("Feature table updated.")

# Define house age feature function
# fe_model.define_feature_function()

# Load data
fe_model.load_data()
logger.info("Data loaded.")

# Perform feature engineering
fe_model.feature_engineering()

# Train the model
fe_model.train()
logger.info("Model training completed.")

# Evaluate model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)
# Drop feature lookup columns and target
test_set = test_set.drop("OverallQual", "GrLivArea", "GarageCars")

model_improved = fe_model.model_improved(test_set=test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

if model_improved:
    # Register the model
    latest_version = fe_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
