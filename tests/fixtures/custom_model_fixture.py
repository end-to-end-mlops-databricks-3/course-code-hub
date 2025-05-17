"""Data fixture."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from conftest import CATALOG_DIR, MLRUNS_DIR
from loguru import logger
from pyspark.sql import SparkSession

from house_price import PROJECT_DIR
from house_price.config import ProjectConfig, Tags
from house_price.models.custom_model import CustomModel

whl_file_name = None  # Global variable to store the .whl file name


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")


@pytest.fixture(scope="session", autouse=True)
def create_mlruns_directory() -> None:
    """Create or recreate the MLFlow tracking directory.

    This fixture ensures that the MLFlow tracking directory is clean and ready for use
    before each test session.
    """
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
        MLRUNS_DIR.mkdir()
        logger.info(f"Created {MLRUNS_DIR} directory for MLFlow tracking")
    else:
        logger.info(f"MLFlow tracking directory {MLRUNS_DIR} does not exist")


@pytest.fixture(scope="session", autouse=True)
def build_whl_file() -> None:
    """Session-scoped fixture to build a .whl file for the project.

    This fixture ensures that the project's distribution directory is cleaned,
    the build process is executed, and the resulting .whl file is identified.
    The fixture runs automatically once per test session.

    :raises RuntimeError: If an unexpected error occurs during the build process.
    :raises FileNotFoundError: If the dist directory or .whl file is not found.
    """
    global whl_file_name
    dist_directory_path = PROJECT_DIR / "dist"
    original_directory = Path.cwd()  # Save the current working directory

    try:
        # Clean up the dist directory if it exists
        if dist_directory_path.exists():
            shutil.rmtree(dist_directory_path)

        # Change to project directory and execute 'uv build'
        os.chdir(PROJECT_DIR)
        subprocess.run(["uv", "build"], check=True, text=True, capture_output=True)

        # Ensure the dist directory exists after the build
        if not dist_directory_path.exists():
            raise FileNotFoundError(f"Dist directory does not exist: {dist_directory_path}")

        # Get list of files in the dist directory
        files = [entry.name for entry in dist_directory_path.iterdir() if entry.is_file()]

        # Find the first .whl file
        whl_file = next((file for file in files if file.endswith(".whl")), None)
        if not whl_file:
            raise FileNotFoundError("No .whl file found in the dist directory.")

        # Set the global variable with the .whl file name
        whl_file_name = whl_file

    except Exception as err:
        raise RuntimeError(f"Unexpected error occurred: {err}") from err

    finally:
        # Restore the original working directory
        os.chdir(original_directory)


@pytest.fixture(scope="function")
def mock_custom_model(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> CustomModel:
    """Fixture that provides a CustomModel instance with mocked Spark interactions.

    Initializes the model with test data and mocks Spark DataFrame conversions to pandas.

    :param config: Project configuration parameters
    :param tags: Tagging metadata for model tracking
    :param spark_session: Spark session instance for testing environment
    :return: Configured CustomModel instance with mocked Spark interactions
    """
    instance = CustomModel(
        config=config,
        tags=tags,
        spark=spark_session,
        code_paths=[f"{PROJECT_DIR.as_posix()}/dist/{whl_file_name}"],
    )

    train_data = pd.read_csv((CATALOG_DIR / "train_set.csv").as_posix())
    # Important Note: Replace NaN with None in Pandas
    train_data = train_data.where(train_data.notna(), None)  # noqa

    test_data = pd.read_csv((CATALOG_DIR / "test_set.csv").as_posix())
    test_data = test_data.where(test_data.notna(), None)  # noqa

    ## Mock Spark interactions
    # Mock Spark DataFrame with toPandas() method
    mock_spark_df_train = MagicMock()
    mock_spark_df_train.toPandas.return_value = train_data
    mock_spark_df_test = MagicMock()
    mock_spark_df_test.toPandas.return_value = test_data

    # Mock spark.table method
    mock_spark = MagicMock()
    mock_spark.table.side_effect = [mock_spark_df_train, mock_spark_df_test]
    instance.spark = mock_spark

    return instance
