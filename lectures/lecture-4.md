---
layout: default
title: Lecture 4 - Model Serving Endpoints
---

# Lecture 4: Model Serving Endpoints

## Overview

Last week we demonstrated model training and registering for different use cases. This week, we show three different serving endpoint creations for different scenarios: Feature serving, model serving, and model serving with feature lookup.

## Code Structure and Implementations

### 1. Model Serving

**`src/house_price/serving/model_serving.py`**  
**`notebooks/week4.01_deploy_model_serving_endpoint.py`**

Model serving is a process of creating a model serving endpoint that can be used for inference. Endpoint creation process is similar to feature serving, with the exception that we don't need to create a feature table. Instead, we simply create a model serving endpoint that relies on the model we trained.

**Steps:**

- We start with loading the trained and registered model
- Then we create a model serving endpoint using the model. It's important to note that entity name we pass is a registered model name and the version is an existing model version
- We also show an example of traffic split, which is a feature of model serving that allows us to split traffic between multiple model versions
- Finally, we invoke the endpoint and get the predictions. The payload should be a JSON object that includes the same features used for training and values. We need to provide all the features required for prediction
- We also added an example piece of code for simple load test to get average latency

### 2. Model Serving with Feature Lookup

**`src/house_price/serving/fe_model_serving.py`**  
**`notebooks/week4.02_deploy_fe_model_serving_endpoint.py`**

This is a combination of the previous two examples. We load a pre-trained model and create a feature table for lookup. Then we create a model serving endpoint that uses the feature table. Last week, we trained a model with feature lookup and feature func. Now we will create a serving endpoint for that model.

**Steps:**

- We start with creating an online table for existing offline feature table, `house_features`. This is the table we created last week on `week 2 - 05.log_and_register_fe_model.py` notebook
- This online table is required for our model to look up features at serving endpoint
- Next is the same as in the previous notebook, we create an endpoint using the model we registered in the same notebook `week 2 - 05.log_and_register_fe_model.py`. This is the model we registered using feature lookup and feature func
- When we send request to the model endpoint, this time, we won't need to provide all the features. 3 features will be taken from the feature lookup table, also one feature "house_age" will be calculated by the feature function

### 3. Feature Serving

**`src/house_price/serving/feature_serving.py`**  
**`notebooks/week4.03_deploy_feature_serving_endpoint.py`**

**Steps:**

- The process begins by loading both the training and testing datasets, which are then concatenated into a single DataFrame. Subsequently, we load a pre-registered Scikit-learn model for generating predictions and select the features to be used for serving
- Using the loaded model, we generate predictions for our dataset, resulting in a final DataFrame that includes 4 features, one of which is the predicted column. This DataFrame is then utilized to create a feature table in the form of a Delta table
- Next, we establish an online feature table by using the previously created offline feature table as the source, which is also a Delta table. This setup enables the creation of an online table that relies on the feature Delta table crafted in the preceding steps
- To create serving endpoints, it's essential to create a feature spec based on the feature table. This specification defines the source feature Delta table, allowing the feature spec to support both offline and online feature serving

## Notebooks
- [`week4.01_deploy_model_serving_endpoint.py`](../notebooks/week4.01_deploy_model_serving_endpoint.py) - Basic model serving setup
- [`week4.02_deploy_fe_model_serving_endpoint.py`](../notebooks/week4.02_deploy_fe_model_serving_endpoint.py) - Feature engineering model serving
- [`week4.03_deploy_feature_serving_endpoint.py`](../notebooks/week4.03_deploy_feature_serving_endpoint.py) - Feature serving endpoints
- [`week4.04_deploy_db_model_serving_endpoint.py`](../notebooks/week4.04_deploy_db_model_serving_endpoint.py) - Databricks-specific model serving

## Key Concepts

### Serving Patterns
1. **Synchronous**: Real-time API calls
2. **Asynchronous**: Batch processing
3. **Streaming**: Continuous data processing
4. **Edge**: Local device inference

### Endpoint Types
- **Model Serving**: Direct model inference
- **Feature Serving**: Feature computation and serving
- **Composite**: Combined feature and model serving

### Performance Metrics
- **Latency**: Response time (P50, P95, P99)
- **Throughput**: Requests per second
- **Availability**: Uptime percentage
- **Error Rate**: Failed requests percentage

## Deliverable
Deploy your trained model to production:
1. Create model serving endpoint
2. Configure endpoint settings (compute, scaling)
3. Test endpoint with sample requests
4. Implement error handling
5. Set up basic monitoring

## Best Practices
- **Version Management**: Use model versions for deployments
- **Testing**: Thoroughly test before production
- **Monitoring**: Track performance and errors
- **Documentation**: Document API specifications
- **Security**: Implement proper authentication

## Common Challenges
- **Cold Start**: Initial request latency
- **Resource Management**: Balancing cost and performance
- **Model Size**: Large models and memory constraints
- **Dependencies**: Managing model dependencies

## Resources
- [Databricks Model Serving Documentation](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [MLflow Model Deployment](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)

---

**Previous**: [Lecture 3 - Feature Engineering](lecture-3.md) | **Next**: [Lecture 5 - Advanced Serving](lecture-5.md)

[← Back to Course Overview](../index.md)
