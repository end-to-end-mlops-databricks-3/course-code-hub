"""FeaturLookUp Serving module."""

import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
from databricks.sdk.service import catalog

from house_price.config import ProjectConfig


class FeatureLookupServing:
    """Manage Feature Lookup Serving operations."""

    def __init__(self, model_name: str, endpoint_name: str, feature_table_name: str) -> None:
        """Initialize the Feature Lookup Serving Manager.

        :param model_name: Name of the model
        :param endpoint_name: Name of the endpoint
        :param feature_table_name: Name of the feature table
        """
        self.workspace = WorkspaceClient()
        self.feature_table_name = feature_table_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.model_name = model_name
        self.endpoint_name = endpoint_name

    def get_latest_model_version(self) -> str:
        """Get the latest version of the model.

        :return: Latest model version
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def create_or_update_online_table(self) -> None:
        """Create or update an online table for house features."""
        try:
            existing_table = self.workspace.online_tables.get(self.online_table_name)
            logger.info("Online table already exists. Inititating table update.")
            pipeline_id = existing_table.spec.pipeliine_id
            update_response = self.workspace.pipelines.start_update(
                pipeline_id=pipeline_id, full_refresh=False)
            while True:
                update_info = self.workspace.pipelines.get_update(
                    pipeline_id=pipeline_id, update_id=update_response.update_id
                )
                state = update_info.update.state.value

                if state == "COMPLETED":
                    logger.info("Pipeline update completed successfully.")
                    break
                elif state in ["FAILED", "CANCELED"]:
                    logger.error("Pipeline update failed.")
                    raise SystemError("Online table failed to update.")
                elif state == "WAITING_FOR_RESOURCES":
                    logger.warning("Pipeline is waiting for resources.")
                else:
                    logger.info(f"Pipeline is in {state} state.")
                time.sleep(30)
        except catalog.NotFound:
            spec = OnlineTableSpec(
                primary_key_columns=["Id"],
                source_table_full_name=self.feature_table_name,
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
                perform_full_copy=False,
            )
            self.workspace.online_tables.create(name=self.online_table_name, spec=spec)
            logger.info("Online does not exists. Inititating table creation.")

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        wait: bool = False
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks.

        :param version: Version of the model to deploy
        :param workload_size: Workload size (number of concurrent requests). Default is Small = 4 concurrent requests.
        :param scale_to_zero: If True, endpoint scales to 0 when unused
        :param wait: If True, the job will wait for the endpoint creation/update to finish
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            if wait:
                self.workspace.serving_endpoints.create_and_wait(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
            else:
                self.workspace.serving_endpoints.create(
                    name=self.endpoint_name,
                    config=EndpointCoreConfigInput(
                        served_entities=served_entities,
                    ),
                )
        else:
            if wait:
                self.workspace.serving_endpoints.update_config_and_wait(
                    name=self.endpoint_name,
                    served_entities=served_entities,
                )
            else:
                self.workspace.serving_endpoints.update_config(
                    name=self.endpoint_name,
                    served_entities=served_entities,
                ) 
