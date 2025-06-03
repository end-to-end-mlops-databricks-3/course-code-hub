import yaml
from loguru import logger
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.data_processor import DataProcessor, generate_synthetic_data, generate_test_data
from marvelous.common import create_parser

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
is_test = args.is_test

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv", header=True, inferSchema=True
).toPandas()

if not is_test:
    # Generate synthetic data.
    # This is mimicking a new data arrival. In real world, this would be a new batch of data.
    # df is passed to infer schema
    new_data = generate_synthetic_data(df, num_rows=100)
    logger.info("Synthetic data generated.")
else:
    # Generate synthetic data
    # This is mimicking a new data arrival. This is a valid example for integration testing.
    new_data = generate_test_data(df, num_rows=100)
    logger.info("Test data generated.")

# Initialize DataProcessor
data_processor = DataProcessor(new_data, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
