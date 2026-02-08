import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
import mlflow
import json
from huggingface_hub import (
    HfApi,
    create_repo,
    whoami,
)
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow.sklearn

# Dummy comment
# set a flag whether this is pipeline run or local
pipeline = True
if pipeline:
  # Start MLFlow
  mlflow.set_tracking_uri("http://localhost:5000")
  mlflow.set_experiment("mlops-training-experiment")

# Load train and test split data from HF

# Setting and retrirving token from local OS env is not working for me even
# when I exported then in zshrc, falling back to local file based tokens
# Note: in the GitHub pipeline, it will always use HF_TOKEN from os env
token = os.getenv("HF_TOKEN")
if token is None:
    print("HF_TOKEN is None on local setup (os env), falling back to token file")
    with open("tokens.json", "r") as f:
        jdata = json.load(f)
        token = jdata["HF_TOKEN"]

# init api client
api = HfApi(token=token)

# Define constants for the dataset and output paths
Xtrain_PATH = "hf://datasets/sagarsb/tourism-prediction/Xtrain.csv"
Xtest_PATH = "hf://datasets/sagarsb/tourism-prediction/Xtest.csv"
ytrain_PATH = "hf://datasets/sagarsb/tourism-prediction/ytrain.csv"
ytest_PATH = "hf://datasets/sagarsb/tourism-prediction/ytest.csv"

# load files
Xtrain = pd.read_csv(Xtrain_PATH)
Xtest = pd.read_csv(Xtest_PATH)
ytrain = pd.read_csv(ytrain_PATH)
ytest = pd.read_csv(ytest_PATH)

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f'class_weight: {class_weight}')

numeric_features = [
    'Age',
    'CityTier', 
    'DurationOfPitch', 
    'NumberOfPersonVisiting', 
    'NumberOfFollowups',
    'PreferredPropertyStar', 
    'NumberOfTrips', 
    'Passport',
    'PitchSatisfactionScore', 
    'OwnCar', 
    'NumberOfChildrenVisiting', 
    'MonthlyIncome'
  ]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    print("Starting the model building..")
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })
    print("Model building completed..")

    if pipeline:
        print("It is a pipelime run, upload the best model to HuggingFace")
        # Save the model locally
        model_path = "best_tourism_prediction_model_v1.joblib"
        joblib.dump(best_model, model_path)
    
        # Log the model artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        print(f"Model saved as artifact at: {model_path}")
        
        # Repo details
        repo_id = "sagarsb/tourism-prediction"
        repo_type = "model" 
        
        # Check if repo exists; create if missing
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"Repo '{repo_id}' already exists. Using it.")
        except RepositoryNotFoundError:
            print(f"Repo '{repo_id}' not found. Creating...")
            create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=False,
                token=token
            )
            print(f"Repo '{repo_id}' created successfully.")
        
        # Verify authentication
        from huggingface_hub import whoami
        print("Authenticated as:", whoami(token=token)["name"])
        
        # upload the best model
        api.upload_file(
            path_or_fileobj="best_tourism_prediction_model_v1.joblib",
            path_in_repo="best_tourism_prediction_model_v1.joblib",
            repo_id=repo_id,
            repo_type=repo_type,
        )
