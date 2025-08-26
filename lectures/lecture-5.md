---
layout: default
title: Lecture 5 - Databricks Workflows & MLOps Pipeline
---

# Lecture 5: Databricks Workflows & MLOps Pipeline

## Overview

We implement a house price prediction pipeline using Databricks workflows, with different steps for data preprocessing, model training, evaluation, and deployment. Each task is executed sequentially, and certain conditions determine whether a task will trigger the next. In this lecture, we used the model with feature lookup.

## Code Structure and Implementations

### 1. Data Ingestion and Updating Tables

**`01.preprocess_data.py`**

**Description**: Handles data ingestion, filtering new records, and updates train/test datasets.

**Key Steps:**

1. Loads the source dataset and retrieves recent records by comparing timestamps
2. Splits new data into train and test sets (80-20 split)
3. Appends the processed train and test data to existing tables
4. Updates the feature table with the latest data for serving
5. Triggers an online feature refresh pipeline and monitors its completion
6. Sets task values to indicate whether new data was processed

**Purpose**: Ensures the training and test datasets are up-to-date and that the feature table is refreshed with the latest values.

### 2. Model Training

**`02.train_register_fe_model.py`**

**Description**: Trains a LightGBM model on the house price data with engineered features.

**Key Steps:**

1. Loads the train and test datasets from Databricks
2. Performs feature engineering using the Databricks Feature Store, including calculating house age
3. Creates a training pipeline with LightGBM regressor
4. Tracks the training process and parameters in MLflow, logging metrics, artifacts, and model parameters

**Purpose**: Builds and logs a new model for house prices using feature-engineered data.

### 3. Model Evaluation

Evaluates the new model and compares it against the currently deployed model.

**Key Steps:**

1. Loads test data and applies feature engineering
2. Generates predictions using both the new and existing models
3. Calculates performance metrics, specifically Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
4. Compares metrics and decides whether to register the new model if it performs better
5. Sets task values to communicate results for downstream steps

**Purpose**: Ensures the new model performs better than the current model before it can be registered for production use.

### 4. Model Deployment

**`03.deploy_model.py`**

## Key Concepts

### Feature Serving Architecture
- **Online Store**: Low-latency feature retrieval
- **Offline Store**: Batch feature computation
- **Streaming**: Real-time feature updates
- **Hybrid**: Combined online/offline patterns

### Optimization Techniques
1. **Model Quantization**: Reduce model size
2. **Batch Processing**: Group requests for efficiency
3. **Caching**: Store frequently accessed data
4. **Async Processing**: Non-blocking operations

### Deployment Strategies
- **Rolling Updates**: Gradual model replacement
- **Shadow Mode**: Parallel testing without impact
- **Feature Flags**: Dynamic model switching
- **Circuit Breakers**: Fallback mechanisms

## Deliverable
Implement advanced serving setup:
1. Deploy feature serving endpoint
2. Implement caching strategy
3. Set up multi-model serving
4. Create monitoring dashboard
5. Implement fallback mechanisms

## Best Practices
- **Gradual Rollouts**: Test with small traffic percentage
- **Monitoring**: Comprehensive metrics and alerting
- **Fallbacks**: Always have backup plans
- **Documentation**: Clear operational procedures
- **Testing**: Automated testing pipelines

## Advanced Topics
- **Model Compression**: Techniques to reduce model size
- **Edge Deployment**: Serving models on edge devices
- **Federated Learning**: Distributed model training
- **AutoML Integration**: Automated model selection

## Resources
- [Advanced MLOps Patterns](https://ml-ops.org/content/mlops-principles)
- [Feature Store Best Practices](https://www.tecton.ai/blog/what-is-a-feature-store/)

---

**Previous**: [Lecture 4 - Model Serving](lecture-4.md) | **Next**: [Lecture 6 - A/B Testing & Monitoring](lecture-6.md)

[← Back to Course Overview](../index.md)
