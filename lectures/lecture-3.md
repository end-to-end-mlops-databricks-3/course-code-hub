---
layout: default
title: Lecture 3 - Feature Engineering & DynamoDB Integration
---

# Lecture 3: Feature Engineering & DynamoDB Integration

## Overview

In this lecture we covered Databricks feature engineering and DynamoDB integration. Last week we explored 2 different use cases (implementations) that demonstrate basic and complex scenarios for registering models with MLflow. This week we covered a new model with feature lookup which leverages Databricks feature engineering client.

## Code Structure and Implementations

### 1. Model with Feature Lookup

**`notebooks/week3.train_register_fe_model.py`**

In this example, we are using a house price dataset, which consists of static features. Since the features in this dataset don't change over time (e.g., the number of rooms, house size, or location), it wouldn't naturally fit into a real-time feature lookup scenario. However, we still mimic the feature lookup implementation to explain how it would work in practice.

This way, you can still learn the process of registering and logging models with feature lookup, which is useful for more dynamic scenarios, like demand forecasting or real-time recommendation systems, where features need to be fetched at inference time.

### 2. Feature Engineering with DynamoDB

**`notebooks/week3.feature_engineering_demo.py`**

This notebook demonstrates how to use the Feature Engineering Client to:

- **Create a feature table**
- **Define a feature function** 
- **Generate a trainset**
- **Register a model**

In the second part, we provide an alternative approach using DynamoDB. We show how to register a model that performs a DynamoDB lookup at inference time, using an MLflow pyfunc model.

## Key Concepts

### Feature Engineering Client Benefits
- **Real-time Feature Lookup**: Fetch features dynamically at inference time
- **Feature Consistency**: Same features for training and serving
- **Scalability**: Handle large-scale feature serving
- **Integration**: Seamless integration with Databricks ecosystem

### DynamoDB Integration
- **External Feature Store**: Use DynamoDB as an external feature store
- **Custom Lookups**: Implement custom feature lookup logic
- **Flexibility**: Support for various data sources and formats
- **Performance**: Optimized for low-latency feature serving

## Notebooks
- [`week3. feature_engineering_demo.py`](../notebooks/week3.%20feature_engineering_demo.py) - Comprehensive feature engineering examples
- [`week3.train_register_fe_model.py`](../notebooks/week3.train_register_fe_model.py) - Training models with engineered features

## Key Concepts

### Feature Engineering Techniques
1. **Encoding**: One-hot, label, target encoding
2. **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
3. **Binning**: Equal-width, equal-frequency binning
4. **Interactions**: Feature crosses and combinations
5. **Temporal**: Lag features, rolling statistics

### Feature Store Benefits
- **Reusability**: Share features across teams and projects
- **Consistency**: Same features for training and serving
- **Governance**: Feature lineage and documentation
- **Performance**: Optimized feature serving

### Best Practices
- Feature documentation and metadata
- Feature validation and testing
- Monitoring feature quality
- Version control for feature definitions

## Deliverable
Implement comprehensive feature engineering for your dataset:
1. Create multiple feature transformations
2. Implement feature selection
3. Set up feature validation
4. Register features in feature store
5. Train model using engineered features

## Common Pitfalls
- **Data leakage**: Using future information
- **Inconsistent transformations**: Training vs serving differences
- **Over-engineering**: Creating too many irrelevant features
- **Missing validation**: Not checking feature quality

## Resources
- [Databricks Feature Store Documentation](https://docs.databricks.com/machine-learning/feature-store/index.html)
- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/data-prep/construct/collect)

---

**Previous**: [Lecture 2 - Experiment Tracking](lecture-2.md) | **Next**: [Lecture 4 - Model Serving](lecture-4.md)

[← Back to Course Overview](../index.md)
