"""Model monitoring module."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

from house_price.config import ProjectConfig


def create_or_refresh_monitoring(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create or refresh a monitoring table for model serving data.

    This function processes the inference data from a Delta table,
    parses the request and response JSON fields, joins with test and inference sets,
    and writes the resulting DataFrame to a Delta table for monitoring purposes.

    :param config: Configuration object containing catalog and schema names.
    :param spark: Spark session used for executing SQL queries and transformations.
    :param workspace: Workspace object used for managing quality monitors.
    """
    inf_table = spark.sql(
        f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`model-serving-fe_payload`"
    )

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("LotFrontage", DoubleType(), True),
                            StructField("LotArea", IntegerType(), True),
                            StructField("OverallCond", IntegerType(), True),
                            StructField("YearBuilt", IntegerType(), True),
                            StructField("YearRemodAdd", IntegerType(), True),
                            StructField("MasVnrArea", DoubleType(), True),
                            StructField("TotalBsmtSF", IntegerType(), True),
                            StructField("MSZoning", StringType(), True),
                            StructField("Street", StringType(), True),
                            StructField("Alley", StringType(), True),
                            StructField("LotShape", StringType(), True),
                            StructField("LandContour", StringType(), True),
                            StructField("Neighborhood", StringType(), True),
                            StructField("Condition1", StringType(), True),
                            StructField("BldgType", StringType(), True),
                            StructField("HouseStyle", StringType(), True),
                            StructField("RoofStyle", StringType(), True),
                            StructField("Exterior1st", StringType(), True),
                            StructField("Exterior2nd", StringType(), True),
                            StructField("MasVnrType", StringType(), True),
                            StructField("Foundation", StringType(), True),
                            StructField("Heating", StringType(), True),
                            StructField("CentralAir", StringType(), True),
                            StructField("SaleType", StringType(), True),
                            StructField("SaleCondition", StringType(), True),
                            StructField("Id", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.withColumn(
        "timestamp_ms", (F.col("request_time").cast("long") * 1000)
    ).select(
        F.col("request_time").alias("timestamp"),  # Use request_time as the timestamp
        F.col("timestamp_ms"),  # Select the newly created timestamp_ms column
        "databricks_request_id",
        "execution_duration_ms",
        F.col("record.Id").alias("Id"),
        F.col("record.LotFrontage").alias("LotFrontage"),
        F.col("record.LotArea").alias("LotArea"),
        F.col("record.OverallCond").alias("OverallCond"),
        F.col("record.YearBuilt").alias("YearBuilt"),
        F.col("record.YearRemodAdd").alias("YearRemodAdd"),
        F.col("record.MasVnrArea").alias("MasVnrArea"),
        F.col("record.TotalBsmtSF").alias("TotalBsmtSF"),
        F.col("record.MSZoning").alias("MSZoning"),
        F.col("record.Street").alias("Street"),
        F.col("record.Alley").alias("Alley"),
        F.col("record.LotShape").alias("LotShape"),
        F.col("record.LandContour").alias("LandContour"),
        F.col("record.Neighborhood").alias("Neighborhood"),
        F.col("record.Condition1").alias("Condition1"),
        F.col("record.BldgType").alias("BldgType"),
        F.col("record.HouseStyle").alias("HouseStyle"),
        F.col("record.RoofStyle").alias("RoofStyle"),
        F.col("record.Exterior1st").alias("Exterior1st"),
        F.col("record.Exterior2nd").alias("Exterior2nd"),
        F.col("record.MasVnrType").alias("MasVnrType"),
        F.col("record.Foundation").alias("Foundation"),
        F.col("record.Heating").alias("Heating"),
        F.col("record.CentralAir").alias("CentralAir"),
        F.col("record.SaleType").alias("SaleType"),
        F.col("record.SaleCondition").alias("SaleCondition"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("house-prices-model-fe").alias("model_name"),
    )

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    df_final_with_status = (
        df_final.join(test_set.select("Id", "SalePrice"), on="Id", how="left")
        .withColumnRenamed("SalePrice", "sale_price_test")
        .join(inference_set_skewed.select("Id", "SalePrice"), on="Id", how="left")
        .withColumnRenamed("SalePrice", "sale_price_inference")
        .select("*", F.coalesce(F.col("sale_price_test"), F.col("sale_price_inference")).alias("sale_price"))
        .drop("sale_price_test", "sale_price_inference")
        .withColumn("sale_price", F.col("sale_price").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["sale_price", "prediction"])
    )

    house_features = spark.table(f"{config.catalog_name}.{config.schema_name}.house_features")

    df_final_with_features = df_final_with_status.join(house_features, on="Id", how="left")

    df_final_with_features = df_final_with_features.withColumn("GrLivArea", F.col("GrLivArea").cast("double"))

    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create a new monitoring table for model monitoring.

    This function sets up a monitoring table using the provided configuration,
    SparkSession, and workspace. It also enables Change Data Feed for the table.

    :param config: Configuration object containing catalog and schema names
    :param spark: SparkSession object for executing SQL commands
    :param workspace: Workspace object for creating quality monitors
    """
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="sale_price",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
