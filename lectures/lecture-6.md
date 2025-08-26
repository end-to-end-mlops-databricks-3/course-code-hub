---
layout: default
title: Lecture 6 - Model Monitoring & Drift Detection
---

# Lecture 6: Model Monitoring & Drift Detection

## Overview

This lecture covers the implementation of a comprehensive monitoring pipeline for house price prediction models, including drift detection and alerting mechanisms.

## House Price Monitoring Pipeline

Below is an overview of the key scripts and their functionalities for creating a monitoring pipeline for our use case.

## Scripts Overview

### 1. Create Monitoring Data

**`notebooks/week6.01_create_monitoring_table.py`**

This script generates synthetic data for model inference and introduces intentional data drift to see drift monitoring.

#### Key Features:

**Data Loading:**
- Loads existing train and test datasets from the catalog
- Uses a Random Forest model to identify the most important features influencing house prices

**Synthetic Data Generation:**
- Creates two types of synthetic datasets:
  - **Skewed dataset**: Introduces data drift by modifying features like OverallQual and GrLivArea
- Generates unique IDs and adds UTC timestamps to the data

**Data Storage and Feature Updates:**
- Saves synthetic dataset to the tables: `inference_set_skewed`
- Updates the feature store table `house_features` with new synthetic records
- Triggers a pipeline update to refresh the online feature store

**Endpoint Testing:**
- Sends requests to the endpoint in two phases:
  1. **Test Set**: Simulates normal conditions
  2. **Skewed Data**: Mimics data drift over a period of 20-30 minutes

### 2. Create Monitoring Alerts

**`notebooks/week6.02_create_alert.py`**

## Notebooks
- [`week4.05_ab_testing.py`](../notebooks/week4.05_ab_testing.py) - A/B testing implementation and analysis

## Key Concepts

### A/B Testing Components
1. **Control Group**: Baseline model performance
2. **Treatment Group**: New model variant
3. **Success Metrics**: Business and technical KPIs
4. **Statistical Tests**: Significance testing methods

### Monitoring Dimensions
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Data Quality**: Missing values, outliers, schema changes
- **Infrastructure**: Latency, throughput, error rates
- **Business Impact**: Revenue, conversion, user satisfaction

### Drift Types
- **Data Drift**: Input data distribution changes
- **Concept Drift**: Target variable relationship changes
- **Model Drift**: Model performance degradation
- **Prediction Drift**: Output distribution changes

## A/B Testing Framework

### Experiment Design
1. **Hypothesis Formation**: Clear, testable hypotheses
2. **Metric Selection**: Primary and secondary metrics
3. **Sample Size Calculation**: Statistical power analysis
4. **Randomization Strategy**: User assignment methods

### Implementation Steps
1. **Traffic Splitting**: Route users to different models
2. **Data Collection**: Track all relevant metrics
3. **Statistical Analysis**: Test significance and effect size
4. **Decision Making**: Go/no-go based on results

## Monitoring Setup

### Key Metrics to Track
- **Accuracy Metrics**: Model performance over time
- **Latency Metrics**: Response time percentiles
- **Volume Metrics**: Request rates and patterns
- **Error Metrics**: Error rates and types

### Alerting Strategy
- **Threshold-based**: Simple metric thresholds
- **Anomaly Detection**: Statistical anomaly identification
- **Trend Analysis**: Performance trend monitoring
- **Composite Alerts**: Multiple metric combinations

## Deliverable
Implement comprehensive monitoring and A/B testing:
1. Set up A/B test between two model versions
2. Implement monitoring dashboard
3. Configure alerting system
4. Run statistical analysis of results
5. Document decision-making process

## Best Practices
- **Clear Hypotheses**: Define what success looks like
- **Sufficient Sample Size**: Ensure statistical power
- **Multiple Metrics**: Don't rely on single metric
- **Long-term Tracking**: Monitor beyond initial deployment
- **Documentation**: Record all decisions and learnings

## Common Pitfalls
- **Peeking**: Checking results too early
- **Multiple Testing**: Not correcting for multiple comparisons
- **Selection Bias**: Non-random user assignment
- **Confounding Variables**: External factors affecting results

## Advanced Topics
- **Multi-armed Bandits**: Dynamic traffic allocation
- **Bayesian A/B Testing**: Probabilistic approach
- **Causal Inference**: Understanding true model impact
- **Continuous Experimentation**: Always-on testing culture

## Resources
- [A/B Testing Best Practices](https://exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf)
- [Model Monitoring Guide](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)
- [Statistical Significance Calculator](https://www.evanmiller.org/ab-testing/sample-size.html)

---

**Previous**: [Lecture 5 - Advanced Serving](lecture-5.md)

[← Back to Course Overview](../index.md)
