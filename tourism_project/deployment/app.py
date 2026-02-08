import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="sagarsb/tourism-prediction", filename="best_tourism_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for 'Visit with Us' staff that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them, based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to opt for tourism package.")

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

Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier = st.number_input("City Tier (The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3))", min_value=1, max_value=3, value=1)
DurationOfPitch = st.number_input("Pitch Duration (in minutes - sales pitch delivered to the customer)", min_value=1, max_value=200, value=30)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting (Total number of people accompanying the customer on the trip.)", min_value=0, max_value=10, value=1)
NumberOfFollowups = st.number_input("Number Of Followups (Total number of follow-ups by the salesperson after the sales pitch)", min_value=0, max_value=20, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star (Preferred hotel rating by the customer.)", min_value=1, max_value=7, value=4)
NumberOfTrips = st.number_input("Number Of Trips (Average number of trips the customer takes annually)", min_value=0, max_value=100, value=5)
Passport = st.selectbox("Passport (Whether the customer holds a valid passport", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (Score indicating the customer's satisfaction with the sales pitch.)", min_value=1, max_value=5, value=5)
OwnCar = st.selectbox("Own Car (Whether the customer owns a car (0: No, 1: Yes))", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting (Number of children below age 5 accompanying the customer)", min_value=0, max_value=10, value=2)
MonthlyIncome = st.number_input("Monthly Income (Gross monthly income of the customer)", min_value=500, max_value=200000, value=5000)

TypeofContact = st.selectbox("Type of Contact (The method by which the customer was contacted)", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation (country where the customer resides)", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender (Gender of the customer)", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched (The type of product pitched to the customer.)", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status (Marital status of the customer)", ["Single", "Divorced", "Married"])
Designation = st.selectbox("Designation (Customer's designation in their current organization.)", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier, 
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting, 
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar, 
    'NumberOfTrips': NumberOfTrips, 
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore, 
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting, 
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.5

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "opt" if prediction == 1 else "not opt"
    st.write(f"Based on the information provided, the customer is likely to {result} for tourism package.")
