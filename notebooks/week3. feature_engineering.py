# Databricks notebook source

from pyspark.sql import SparkSession
import mlflow

from house_price.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from marvelous.common import is_databricks
from dotenv import load_dotenv
import os
from mlflow import MlflowClient
import pandas as pd
from house_price import __version__
from mlflow.utils.environment import _mlflow_conda_env
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup


# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# COMMAND ----------
# create feature table with information about houses
# Option 1: feature engineering client
feature_table_name = f"{config.catalog_name}.{config.schema_name}.house_features"
lookup_features = ["OverallQual", "GrLivArea", "GarageCars"]

feature_table = fe.create_table(
   name=feature_table_name,
   primary_keys=["ID"],
   df=train_set[lookup_features],
   description="House features table",
)

spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

fe.write_table(
   name=feature_table_name,
   df=test_set[lookup_features],
   mode="merge",
)

# COMMAND ----------
# create feature table with information about houses
# Option 2: SQL commands
spark.sql(f"""
          CREATE OR REPLACE TABLE {feature_table_name}
          (Id STRING NOT NULL, OverallQual INT, GrLivArea INT, GarageCars INT);
          """)
# primary key on Databricks is not enforced!
spark.sql(f"ALTER TABLE {feature_table_name} ADD CONSTRAINT house_pk PRIMARY KEY(Id);")
spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT Id, OverallQual, GrLivArea, GarageCars
          FROM {config.catalog_name}.{config.schema_name}.train_set
          """)
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT Id, OverallQual, GrLivArea, GarageCars
          FROM {config.catalog_name}.{config.schema_name}.test_set
          """)

# COMMAND ----------
# create feature function
# docs: https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function

# problems with feature functions:
# functions are not versioned 
# functions may behave differently depending on the runtime (and version of packages and python)
# there is no way to enforce python version & package versions for the function 
# this is only supported from runtime 17
# advised to use only for simple calculations

# Option 1: with Python

function_name = f"{config.catalog_name}.{config.schema_name}.calculate_house_age"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {function_name}(year_built INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime
        return datetime.now().year - year_built
        $$
        """)

# COMMAND ----------
# it is possible to define simple functions in sql only without python
# Option 2
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {function_name}(year_built INT)
        RETURNS INT
        RETURN year(current_date()) - year_built;
        """)

# COMMAND ----------
# execute function
spark.sql(f"SELECT {function_name}(1960);")

# COMMAND ----------
# create a training set
training_set = fe.create_training_set(
    df=train_set,
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["OverallQual", "GrLivArea", "GarageCars"],
            lookup_key="Id",
                ),
        FeatureFunction(
            udf_name=function_name,
            output_name="house_age",
            input_bindings={"year_built": "YearBuilt"},
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------
# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.num_features + config.cat_features + ["house_age"]]
y_train = training_df[config.target]

# COMMAND ----------

pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                           config.cat_features)],
            remainder="passthrough")
            ),
               ("regressor", LGBMRegressor(**config.parameters))]
        )

pipeline.fit(X_train, y_train)

# COMMAND ----------
mlflow.set_experiment("/Shared/demo-model-fe")
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
    
# COMMAND ----------
model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------
# make predictions
features = [f for f in "Id" + config.num_features + config.cat_features if f not in lookup_features]
predictions = fe.score_batch(
    f"models:/{model_name}/{model_version.version}",
    test_set[features]
)

# COMMAND ----------
# make predictions for a non-existing entry -> error!


# COMMAND ----------
# what if we want to replace with a default value if entry is not found
# what if we want to look up value in another table? the logics get complex
# problems that arize: functions/ lookups always get executed (if statememt is not possible)
# it can get slow...

# step 1: create 3 feature functions

# step 2: redefine create training set

# try again

# COMMAND ----------
# We ran into more limitations when we tried complex data types as output of a feature function
# and then tried to use it for serving
# al alternatve solution: using an external database (we use DynamoDB here)

# create a DynamoDB table
# insert records into dynamo DB & read from dynamoDB

# create a pyfunc model
