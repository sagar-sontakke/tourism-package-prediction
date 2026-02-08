# tourism-package-prediction
A repository for Tourism Package Prediction application and its GitHub and MLOps pipeline

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The workflow is defined in `.github/workflows/pipeline.yaml` and automates the following steps:

1. **register-dataset**: Installs dependencies and uploads the dataset to the Hugging Face Hub using `data_register.py`.
2. **data-prep**: Prepares and cleans the data using `prep.py` after the dataset is registered.
3. **model-traning**: (Typo in workflow, should be `model-training`) Installs dependencies, starts an MLflow server, and runs `train.py` to train and log the model.
4. **deploy-hosting**: Deploys the frontend (Streamlit app and related files) to Hugging Face Spaces using `hosting.py`.

The pipeline is triggered automatically on every push to the `main` branch, ensuring that data, models, and deployments are always up to date.


# Tourism Package Prediction

This repository contains an end-to-end MLOps pipeline for predicting whether a customer will purchase a newly introduced Wellness Tourism Package. The project leverages machine learning, data engineering, and deployment best practices to deliver a robust, production-ready solution.

## Features
- **Data Cleaning & Preparation:** Scripts for cleaning and preparing raw tourism data.
- **Model Training:** Automated training pipeline using scikit-learn and XGBoost, with experiment tracking via MLflow.
- **Model Deployment:** Streamlit web app for real-time predictions, integrated with Hugging Face Hub for model hosting.
- **MLOps Workflow:** Includes scripts for data registration, model versioning, and deployment automation.

## Project Structure
```
tourism_project/
	data/                # Raw and processed data files
	deployment/          # Streamlit app and deployment scripts
	hosting/             # Scripts for uploading to Hugging Face Spaces
	model_building/      # Data prep, training, and registration scripts
README.md              # Project documentation
```

*Further setup, usage, and requirements details will be added in the next steps.*
