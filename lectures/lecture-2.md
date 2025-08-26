---
layout: default
title: Lecture 2 - MLflow Experiment Tracking & Model Registry
---

# Lecture 2: MLflow Experiment Tracking & Model Registry

## Overview

In this lecture we covered MLflow experiment tracking and model registry concepts. We explored two different use cases (implementations) that demonstrate basic and complex scenarios for registering models with MLflow. For Databricks model deployment, we use MLflow for experiment tracking and model registry, which are essential components as mentioned in the first lecture.

## Code Structure and Implementations

### 1. Data Preparation

**`scripts/01.process_data.py`**

Our preprocessing code from the previous lecture with functionality to save test and train set tables in Unity Catalog.

### 2. Logging and Registering a Model

**`notebooks/week2.train_register_basic_model.py`**

**Steps:**

- We start by loading our train and test datasets, then we train an sklearn pipeline within an MLflow run. During this run, we set 2 tags. Tags are a useful feature in MLflow for attaching additional information to your run. For example, we tag the Git commit and branch name to improve traceability.

- **Logging Metrics and Data**: After the run is complete, we log the desired metrics. These metrics are visible in the UI, in Databricks experiments. Additionally, we log the dataset we used for this run to keep track of the data.

- **Model Registration**: Next, we show how to register a model. MLflow creates model artifacts during the run, and by using the model_uri from that specific run, we register the model and obtain its version.

- **Retrieving Data**: Finally, we show how to retrieve the data from the MLflow run, as it was logged during the previous step.

### 3. Logging and Registering Custom Model

**`notebooks/week2.train_register_custom_model.py`**

This is a more advanced use case. In this implementation, we explain two additional aspects:

**Registering a Custom Model**: While MLflow provides built-in flavors for commonly used ML libraries, if you're using your own custom code for training and registering a model, you need to use the pyfunc flavor. We show how to wrap a model for this purpose. In our example, we trained an sklearn model (which isn't custom), but we treat it as if it were a custom model. We assume the trained model is custom and register it using pyfunc. Another reason for using this method could be that you want to control how predictions are returned, rather than using the default output from the library. For instance, we demonstrate returning predictions as a dictionary, `{predictions: [a, b, c, d]}`, instead of a simple list, `[a, b, c, d]`. The wrapper class allows this customization.

**Using a Custom Library**: The second thing we show is how to use a custom Python library in your MLflow run. Suppose you have a custom function for transforming predictions or preprocessing inputs before calling `model.predict`. If this custom functionality is part of a Python module you've developed, you can make it available in your MLflow run by passing it as a pip requirement. In the pyfunc wrapper class, you can then import and use your custom module like any other Python library. In our example, we wanted to use the `adjust_predictions` function after predictions are generated. We packaged our `house_price` module, which contains this function, and provided it as a pip requirement. The path we pass is the location of .whl file of our package in Volumes.

## Notebooks
- [`week2. experiment_tracking.py`](../notebooks/week2.%20experiment_tracking.py) - Basic experiment tracking setup
- [`week2. log_register_model.py`](../notebooks/week2.%20log_register_model.py) - Complete model logging and registration
- [`week2. train_register_basic_model.py`](../notebooks/week2.%20train_register_basic_model.py) - Basic model training and registration
- [`week2. train_register_custom_model.py`](../notebooks/week2.%20train_register_custom_model.py) - Custom model implementation

## Key Concepts

### MLflow Components
1. **Tracking**: Log parameters, metrics, and artifacts
2. **Projects**: Package ML code in reusable format
3. **Models**: Deploy models to various platforms
4. **Registry**: Centralized model store

### Best Practices
- Consistent naming conventions
- Meaningful experiment descriptions
- Regular model evaluation
- Proper artifact organization

## Deliverable
Implement experiment tracking for your dataset:
1. Set up MLflow tracking
2. Train multiple models with different parameters
3. Log all relevant metrics and artifacts
4. Register your best model

## Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Databricks MLflow Guide](https://docs.databricks.com/mlflow/index.html)

---

**Previous**: [Lecture 1 - Introduction](lecture-1.md) | **Next**: [Lecture 3 - Feature Engineering](lecture-3.md)

[← Back to Course Overview](../index.md)
