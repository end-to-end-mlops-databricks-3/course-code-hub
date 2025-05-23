# Databricks notebook source
# install dependencies
%pip install -e ..
%pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

#restart python
%restart_python

# COMMAND ----------

# system path update, must be after %restart_python
# caution! This is not a great approach
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

# A better approach (this file must be present in a notebook folder, achieved via synchronization)
%pip install house_price-1.0.1-py3-none-any.whl

# COMMAND ----------

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
from pyspark.errors import AnalysisException
import numpy as np
from datetime import datetime
import boto3


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

feature_table_name = f"{config.catalog_name}.{config.schema_name}.house_features_demo"
lookup_features = ["OverallQual", "GrLivArea", "GarageCars"]


# COMMAND ----------

# Option 1: feature engineering client
feature_table = fe.create_table(
   name=feature_table_name,
   primary_keys=["Id"],
   df=train_set[["Id"]+lookup_features],
   description="House features table",
)

spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

fe.write_table(
   name=feature_table_name,
   df=test_set[["Id"]+lookup_features],
   mode="merge",
)

# COMMAND ----------

# create feature table with information about houses
# Option 2: SQL

spark.sql(f"""
          CREATE OR REPLACE TABLE {feature_table_name}
          (Id STRING NOT NULL, OverallQual INT, GrLivArea INT, GarageCars INT);
          """)
# primary key on Databricks is not enforced!
try:
    spark.sql(f"ALTER TABLE {feature_table_name} ADD CONSTRAINT house_pk_demo PRIMARY KEY(Id);")
except AnalysisException:
    pass
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

function_name = f"{config.catalog_name}.{config.schema_name}.calculate_house_age_demo"

# COMMAND ----------


# Option 1: with Python
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {function_name}(year_built BIGINT)
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
        CREATE OR REPLACE FUNCTION {function_name}_sql (year_built BIGINT)
        RETURNS INT
        RETURN year(current_date()) - year_built;
        """)

# COMMAND ----------

# execute function
spark.sql(f"SELECT {function_name}_sql(1960) as house_age;")

# COMMAND ----------

# create a training set
training_set = fe.create_training_set(
    df=train_set.drop("OverallQual", "GrLivArea", "GarageCars"),
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
features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set[features]
)

# COMMAND ----------

predictions.select("prediction").show(5)

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "Id",
    (col("Id").cast("long") + 1000000).cast("string")
)

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id
)

# COMMAND ----------

# make predictions for a non-existing entry -> error!
predictions.select("prediction").show(5)

# COMMAND ----------

overallqual_function = f"{config.catalog_name}.{config.schema_name}.replace_overallqual_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {overallqual_function}(OverallQual INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if OverallQual is None:
            return 5
        else:
            return OverallQual
        $$
        """)

grlivarea_function = f"{config.catalog_name}.{config.schema_name}.replace_grlivarea_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {grlivarea_function}(GrLivArea INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if GrLivArea is None:
            return 1000
        else:
            return GrLivArea
        $$
        """)

garagecars_function = f"{config.catalog_name}.{config.schema_name}.replace_garagecars_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {garagecars_function}(GarageCars INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if GarageCars is None:
            return 2
        else:
            return GarageCars
        $$
        """)

# COMMAND ----------

# what if we want to replace with a default value if entry is not found
# what if we want to look up value in another table? the logics get complex
# problems that arize: functions/ lookups always get executed (if statememt is not possible)
# it can get slow...

# step 1: create 3 feature functions

# step 2: redefine create training set

# try again

# create a training set
training_set = fe.create_training_set(
    df=train_set.drop("OverallQual", "GrLivArea", "GarageCars"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["OverallQual", "GrLivArea", "GarageCars"],
            lookup_key="Id",
            rename_outputs={"OverallQual": "lookup_OverallQual",
                            "GrLivArea": "lookup_GrLivArea",
                            "GarageCars": "lookup_GarageCars"}
                ),
        FeatureFunction(
            udf_name=overallqual_function,
            output_name="OverallQual",
            input_bindings={"OverallQual": "lookup_OverallQual"},
            ),
        FeatureFunction(
            udf_name=grlivarea_function,
            output_name="GrLivArea",
            input_bindings={"GrLivArea": "lookup_GrLivArea"},
        ),
        FeatureFunction(
            udf_name=garagecars_function,
            output_name="GarageCars",
            input_bindings={"GarageCars": "lookup_GarageCars"},
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

#pipeline
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
model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "Id",
    (col("Id").cast("long") + 1000000).cast("string")
)

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id
)

# COMMAND ----------

# make predictions for a non-existing entry -> no error!
predictions.select("prediction").show(5)

# COMMAND ----------

import boto3

region_name = "eu-west-1"
aws_access_key_id = os.environ["aws_access_key_id"]
aws_secret_access_key = os.environ["aws_secret_access_key"]

client = boto3.client(
    'dynamodb',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# COMMAND ----------

response = client.create_table(
    TableName='HouseFeatures',
    KeySchema=[
        {
            'AttributeName': 'Id',
            'KeyType': 'HASH'  # Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'Id',
            'AttributeType': 'S'  # String
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Table creation initiated:", response['TableDescription']['TableName'])

# COMMAND ----------

client.put_item(
    TableName='HouseFeatures',
    Item={
        'Id': {'S': 'house_001'},
        'OverallQual': {'N': '8'},
        'GrLivArea': {'N': '2450'},
        'GarageCars': {'N': '2'}
    }
)

# COMMAND ----------

response = client.get_item(
    TableName='HouseFeatures',
    Key={
        'Id': {'S': 'house_001'}
    }
)

# Extract the item from the response
item = response.get('Item')
print(item)

# COMMAND ----------

from itertools import islice

rows = spark.table(feature_table_name).toPandas().to_dict(orient="records")

def to_dynamodb_item(row):
    return {
        'PutRequest': {
            'Item': {
                'Id': {'S': str(row['Id'])},
                'OverallQual': {'N': str(row['OverallQual'])},
                'GrLivArea': {'N': str(row['GrLivArea'])},
                'GarageCars': {'N': str(row['GarageCars'])}
            }
        }
    }

items = [to_dynamodb_item(row) for row in rows]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch in chunks(items, 25):
    response = client.batch_write_item(
        RequestItems={
            'HouseFeatures': batch
        }
    )
    # Handle any unprocessed items if needed
    unprocessed = response.get('UnprocessedItems', {})
    if unprocessed:
        print("Warning: Some items were not processed. Retry logic needed.")

# COMMAND ----------

# We ran into more limitations when we tried complex data types as output of a feature function
# and then tried to use it for serving
# al alternatve solution: using an external database (we use DynamoDB here)

# create a DynamoDB table
# insert records into dynamo DB & read from dynamoDB

# create a pyfunc model

# COMMAND ----------


class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting house prices.
    """

    def __init__(self, model: object) -> None:
        """Initialize the HousePriceModelWrapper.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray
    ) -> dict[str, float]:
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A dictionary containing the adjusted prediction.
        """
        client = boto3.client('dynamodb',
                                   aws_access_key_id=os.environ["aws_access_key_id"],
                                   aws_secret_access_key=os.environ["aws_secret_access_key"],
                                   region_name=os.environ["region_name"])

        parsed = []
        for lookup_id in model_input["Id"]:
            raw_item = client.get_item(
                TableName='HouseFeatures',
                Key={'Id': {'S': lookup_id}})["Item"]
            parsed_dict = {key: int(value['N']) if 'N' in value else value['S']
                      for key, value in raw_item.items()}
            parsed.append(parsed_dict)
        lookup_df=pd.DataFrame(parsed)
        merged_df = model_input.merge(lookup_df, on="Id", how="left").drop("Id", axis=1)

        merged_df["GarageCars"] = merged_df["GarageCars"].fillna(2)
        merged_df["GrLivArea"] = merged_df["GrLivArea"].fillna(1000)
        merged_df["OverallQual"] = merged_df["OverallQual"].fillna(5)
        merged_df["house_age"] = datetime.now().year - merged_df["YearBuilt"]
        predictions = self.model.predict(merged_df)

        return [int(x) for x in predictions]

# COMMAND ----------

custom_model = HousePriceModelWrapper(pipeline)

# COMMAND ----------

features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
data = test_set.select(*features).toPandas()
data

# COMMAND ----------

custom_model.predict(context=None, model_input=data)

# COMMAND ----------

#log model
mlflow.set_experiment("/Shared/demo-model-fe-pyfunc")
with mlflow.start_run(run_name="demo-run-model-fe-pyfunc",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=data, model_output=custom_model.predict(context=None, model_input=data))
    mlflow.pyfunc.log_model(
                python_model=custom_model,
                artifact_path="lightgbm-pipeline-model-fe",
                signature=signature,
            )


# COMMAND ----------

# predict
mlflow.models.predict(f"runs:/{run_id}/lightgbm-pipeline-model-fe", data[0:1])
