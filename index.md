---
layout: default
title: Marvelous MLOps - End-to-End MLOps with Databricks
---

# Marvelous MLOps
## End-to-End MLOps with Databricks Course

Welcome to the comprehensive MLOps course using Databricks! This course covers the complete machine learning operations lifecycle from development to production deployment.

## Course Information

- **Schedule**: Weekly lectures on Wednesdays 16:00-18:00 CET
- **Runtime**: Databricks 15.4 LTS (Python 3.11)
- **Dataset**: [House Price Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) from Kaggle
- **Delivery**: Weekly deliverables with PR-based submissions

## Course Structure

### Week 1: Foundation & Best Practices
- [Lecture 1: Introduction to MLOps](lectures/lecture-1.md)
- Topics: Pretty vs Ugly Notebooks, Code Quality

### Week 2: Experiment Tracking & Model Management
- [Lecture 2: Experiment Tracking](lectures/lecture-2.md)
- Topics: MLflow, Model Registration, Versioning

### Week 3: Feature Engineering
- [Lecture 3: Feature Engineering](lectures/lecture-3.md)
- Topics: Feature Stores, Data Pipelines

### Week 4: Model Deployment
- [Lecture 4: Model Serving](lectures/lecture-4.md)
- Topics: Serving Endpoints, Real-time Inference

### Week 5: Advanced Deployment
- [Lecture 5: Advanced Serving](lectures/lecture-5.md)
- Topics: Feature Serving, Batch Inference

### Week 6: Testing & Monitoring
- [Lecture 6: A/B Testing & Monitoring](lectures/lecture-6.md)
- Topics: A/B Testing, Model Monitoring, Performance Tracking

## Getting Started

### Environment Setup

1. **Install UV** (Python package manager):
   ```bash
   # Follow installation guide: https://docs.astral.sh/uv/getting-started/installation/
   ```

2. **Create Environment**:
   ```bash
   uv venv -p 3.11 .venv
   source .venv/bin/activate
   uv sync --extra dev
   ```

3. **GitHub Authentication** (for private dependencies):
   - Authenticate to GitHub using HTTPS
   - Credentials stored in keychain for local development
   - Databricks uses Course cluster policy with init script

## Resources

- [Course Repository](https://github.com/end-to-end-mlops-databricks-3/course-code-hub)
- [Taskfile Documentation](Taskfiles.md)
- [Project Configuration](project_config.yml)

## Submission Process

1. Create a feature branch for your deliverable
2. Implement solution with your own dataset
3. Create PR to main branch
4. Code review and CI pipeline approval required
5. Deliverables can be submitted with delay but must be completed before final week

---

*Course materials, presentations, and video recordings are shared after each lecture.*
