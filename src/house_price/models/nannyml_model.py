import mlflow
from mlflow import MlflowClient
import nannyml as nml
from nannyml.chunk import SizeBasedChunker
from house_price.config import ProjectConfig, Tags
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

class HousePriceNannyMLWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting house prices.
    """

    def __init__(self, config: ProjectConfig, model: object, spark: SparkSession) -> None:
        """Initialize the nanny model wrapper.

        :param model: The underlying nanny ML model.
        """
        self.cofig = config
        self.spark = spark
        self.model = nml.DLE(
            feature_column_names=config.num_features + config.cat_features,
            y_pred='prediction',
            y_true= config.target,
            timestamp_column_name='timestamp',
            metrics=['rmse'],
            chunk_size=6000,
            tune_hyperparameters=False
        )

    def train(self, reference_data: pd.DataFrame):
        """Train the model and log it to MLflow."""
        # load the test set. we need it to fit the monitoring model.
        # test_set = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.test_set").toPandas()
        # fit the model NannyML model
        self.model.fit(reference_data)
        # Log and register the model
        self.log_and_register_model()

    def log_and_register_model(self):
        client=MlflowClient()
        with mlflow.start_run() as run:
            # Log the model
            mlflow.pyfunc.log_model(
                artifact_path="nannyml_model",
                python_model=self,
            )
            # Register the model
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/nannyml_model",
                name=f"{self.config.catalog}.{self.config.schema_name}.nannyml_model"
            )

            client.set_registered_model_alias(name="NannyHousePriceModel",
                alias="latest-model",
                version = model_version.version)


    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Loads the registered NannyML model from MLflow and runs estimations on model_input.

        :param model_input: DataFrame with input features for estimation.
        :param config: ProjectConfig object with catalog and schema info.
        :return: DataFrame with NannyML estimations.
        """
        model_name = f"{self.config.catalog}.{self.config.schema_name}.nannyml_model"
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@latest-model")
        
        model.chunker = SizeBasedChunker(chunk_size=1)
        predictions = model.estimate(model_input)
        
        return predictions.to_df()