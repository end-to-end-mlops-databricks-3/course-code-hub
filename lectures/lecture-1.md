---
layout: default
title: Lecture 1 - Introduction to MLOps & Development Environment
---

# Lecture 1: Introduction to MLOps & Development Environment

## Course Logistics

- **Weekly lectures**: Wednesdays, 16:00-18:00 CET
- **Pre-Lecture**: Code is shared before each session
- **Post-Lecture**: Slides and materials shared right after; video uploaded within 24 hours
- **Assignments**: Weekly deliverables implemented using your own dataset
- **GitHub Repositories**: Every student has a personal repo under end-to-end-mlops-databricks-3 organization
- **Submission**: Create a feature branch, submit a PR to main, and wait for approval + CI success
- **Deadline**: Final submissions by **June 18th** (Demo Day). Optionally, showcase your project publicly

## Overview

In this lecture, we explained the fundamentals of MLOps, emphasizing its role in ensuring traceability, reproducibility, and monitoring throughout the machine learning lifecycle. We discussed key MLOps principles and essential tools, highlighting how Databricks simplifies end-to-end ML operations.

We then focused on developing in a reproducible environment, which is crucial for seamless production deployment. We explained Databricks clusters and how to configure them.

## Development Tools

We introduced Databricks development tools that streamline local development and ensure consistency:

### Git Repos
Sync and run code directly in the Databricks environment.
- [Databricks Git Repos Documentation](https://docs.databricks.com/en/repos/index.html)

### VS Code Extension
Enables easy file synchronization, leveraging Databricks Connect and Databricks Asset Bundles for an optimized development workflow.
- [VS Code Extension Documentation](https://docs.databricks.com/en/dev-tools/vscode-ext/index.html)
- [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=databricks.databricks)

## From Notebook to Production

See the article: [Bridging the Gap: Converting Data Science Notebooks to Production Code](https://marvelousmlops.substack.com/p/bridging-the-gap-converting-data)

## Databricks CLI & Authentication

### 1. Install CLI

**For macOS:**
```bash
brew tap databricks/tap
brew install databricks
```

**For Windows:**
Download from [GitHub Releases](https://github.com/databricks/cli/releases)

### 2. Authentication
```bash
databricks auth login --configure-cluster --host <workspace-url>
```

**Resources:**
- [Official CLI Documentation](https://docs.databricks.com/en/dev-tools/cli/install.html)
- [Developing on Databricks](https://marvelousmlops.substack.com/p/developing-on-databricks-without)
- [Handy Databricks Features](https://marvelousmlops.substack.com/p/handy-databricks-features-for-development)

## Use Case: Real Estate Price Prediction

**Dataset**: [House Prices (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

We will build **3 ML services**:

1️⃣ **Real-time model**: Price prediction using only user-inputted features  
2️⃣ **Hybrid real-time model**: Some features from the user, some from a database lookup  
3️⃣ **Batch prediction service**: Precomputed prices stored in an online table

### Example Datasets from Previous Cohorts
- [Bank Marketing Dataset](https://www.kaggle.com/datasets/krantiswalke/bankfullcsv)
- [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019)
- [Red Wine Quality](https://www.kaggle.com/code/nimapourmoradi/red-wine-quality)
- [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)

## Notebooks
- [`week1. pretty_notebook.py`](../notebooks/week1.%20pretty_notebook.py) - Example of well-structured ML code
- [`week1. ugly_notebook.py`](../notebooks/week1.%20ugly_notebook.py) - Common anti-patterns to avoid

## Week 1 Assignment

- **Select a dataset** as you wish
- **Start working** on the repository created for you under end-to-end-mlops-databricks-3 organization
- **Prepare a processing module**, similar to the one shown in the lecture, practice coding best practices
- **Create a PR** on your personal repository
- **Submit PR** in pr-submissions channel under Cohort 3 private hub

**Lecture code** (branch week1): [Course Code Hub](https://github.com/end-to-end-mlops-databricks-3/course-code-hub)

## Technical Notes

### File Transfer to Databricks

How to copy files from local to Databricks:

```bash
databricks auth login
databricks fs cp /path/to/source dbfs:/Volumes/path/to/destination
```

[Documentation Link](https://docs.gcp.databricks.com/en/dev-tools/cli/fs-commands.html)

### Private Package Setup

**Step 1**: Update `pyproject.toml` file. Locally, it works without token if you use HTTPS (not SSH) to clone repos. You may need `GITHUB_TOKEN` env var in `.env` file.

**Step 2**: For cluster creation, use Course policy - it will already have env var setup & init scripts.

**Important for UV users**: Do not use `tool.uv.sources` to specify your dependencies. If you build the package using `uv build` and then install the wheel, the source specified there will be ignored. In the best case, the installation will fail. In the worst case, a package with the same name might be available on PyPI and will be installed instead.

## Resources
- [Course Repository Setup](../README.md)
- [Environment Configuration](../pyproject.toml)
- [Pre-commit Hooks](../.pre-commit-config.yaml)

---

**Next**: [Lecture 2 - Experiment Tracking](lecture-2.md)

[← Back to Course Overview](../index.md)
