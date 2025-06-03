"""Feature Serving module."""

from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class FeatureServing:
    """Manages feature serving operations in Databricks."""

    def __init__(self, feature_table_name: str, feature_spec_name: str, endpoint_name: str) -> None:
        """Initialize the FeatureServing instance.

        :param feature_table_name: Name of the feature table
        :param feature_spec_name: Name of the feature specification
        :param endpoint_name: Name of the serving endpoint
        """
        self.feature_table_name = feature_table_name
        self.workspace = WorkspaceClient()
        self.feature_spec_name = feature_spec_name
        self.endpoint_name = endpoint_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.fe = feature_engineering.FeatureEngineeringClient()

    def create_online_table(self) -> None:
        """Create an online table based on the feature table."""
        spec = OnlineTableSpec(
            primary_key_columns=["Id"],  # Feature lookup key
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
                {"triggered": "true"}
            ),  # Sets the policy to update the online table when triggered (not on a schedule)
            perform_full_copy=False,  # Performs incremental updates instead of full snapshot
        )
        self.workspace.online_tables.create(name=self.online_table_name, spec=spec)

    def create_feature_spec(self) -> None:
        """Create a feature spec to enable feature serving."""
        # A configuration for which features to serve.
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="Id",
                feature_names=["GrLivArea", "YearBuilt", "Predicted_SalePrice"],
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features, exclude_columns=None)

    def deploy_or_update_serving_endpoint(self, workload_size: str = "Small", scale_to_zero: bool = True) -> None:
        """Deploy or update the feature serving endpoint in Databricks.

        :param workload_size: Workload size (number of concurrent requests). Default is Small = 4 concurrent requests
        :param scale_to_zero: If True, endpoint scales to 0 when unused
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

        served_entities = served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name, scale_to_zero_enabled=scale_to_zero, workload_size=workload_size
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
