# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Setting and retrirving token from local OS env is not working for me even
# when I exported then in zshrc, falling back to local file based tokens
# Note: in the GitHub pipeline, it will always use HF_TOKEN from os env
token = os.getenv("HF_TOKEN")
if token is None:
    print("HF_TOKEN is None on local setup (os env), falling back to token file")
    with open("tokens.json", "r") as f:
        jdata = json.load(f)
        token = jdata["HF_TOKEN"]

# Define constants for the dataset and output paths
api = HfApi(token=token)
DATASET_PATH = "hf://datasets/sagarsb/tourism-prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Data cleaning
## "Fe male" needs to be renamed to "Female" for "Gender" column
tourism_dataset["Gender"] = tourism_dataset["Gender"].replace({"Fe male": "Female"})

## "Unmarried" needs to be renamed to "Single" for "MaritalStatus" column
tourism_dataset["MaritalStatus"] = tourism_dataset["MaritalStatus"].replace({"Unmarried": "Single"})

# Removing unwanted columns - we will skip below columns
# 1. 'CustomerID' - is a system genearated ID, so skipping it for model training
# 2. 'Unnamed: 0' - this is the auto-generated dataframe column, skipping it

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
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

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

print("Train/test split completed successfully.")

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sagarsb/tourism-prediction",
        repo_type="dataset",
    )
print("Train/test split data successfully uploaded to HugginFace data space.")
