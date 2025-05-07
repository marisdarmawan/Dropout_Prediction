import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# --- Define Custom Transformers (MUST be the same as in training script) ---
# Ensure these classes are identical to those in your training script
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # New features - Ensure column names match your training data
        X['avg_grade'] = (X['Curricular_units_1st_sem_grade'] + X['Curricular_units_2nd_sem_grade']) / 2
        X['avg_approved'] = (X['Curricular_units_1st_sem_approved'] + X['Curricular_units_2nd_sem_approved']) / 2
        # Use .div() and fillna for division by zero/NaNs
        X['approval_rate_1st'] = X['Curricular_units_1st_sem_approved'].div(X['Curricular_units_1st_sem_enrolled'].replace(0, np.nan)).fillna(0)
        X['approval_rate_2nd'] = X['Curricular_units_2nd_sem_approved'].div(X['Curricular_units_2nd_sem_enrolled'].replace(0, np.nan)).fillna(0)
        X['parental_education_avg'] = (X['Fathers_qualification'] + X['Mothers_qualification']) / 2 # Corrected typo Mothers_qualification
        X['parental_occupation_avg'] = (X['Fathers_occupation'] + X['Mothers_occupation']) / 2   # Corrected typo Mothers_occupation
        X['low_income_flag'] = ((X['Scholarship_holder'] == 1) & (X['Tuition_fees_up_to_date'] == 0)).astype(int)
        X['foreign_and_displaced'] = ((X['International'] == 1) | (X['Displaced'] == 1)).astype(int)
        X['no_eval_first_sem'] = (X['Curricular_units_1st_sem_without_evaluations'] > 0).astype(int)
        X['no_eval_second_sem'] = (X['Curricular_units_2nd_sem_without_evaluations'] > 0).astype(int)

        # Handle NaN values by filling with 0 (as in training)
        X.fillna(0, inplace=True)

        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_existing = [col for col in self.cols_to_drop if col in X.columns]
        return X.drop(columns=cols_existing)


# --- Load the trained model and label encoder ---
try:
    pipeline = joblib.load('random_forest_pipeline.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first to generate 'random_forest_pipeline.pkl' and 'label_encoder.pkl'.")
    st.stop() # Stop the app if files are not found

# --- Streamlit App Interface ---
st.title("Student Status Prediction (Dropout/Graduate)")
st.write("Enter student details to predict if they will be a Dropout or a Graduate.")

# Define the list of all original feature columns
original_feature_cols = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
    'Inflation_rate', 'GDP'
]

# Create input widgets for each feature
input_values = {}

st.header("Student Information")
# Group related inputs for better organization in the UI
col1, col2 = st.columns(2)
with col1:
    input_values['Marital_status'] = st.number_input("Marital Status", min_value=0, value=1, help="E.g., 1: Single, 2: Married")
    input_values['Application_mode'] = st.number_input("Application Mode", min_value=0, value=1, help="Code for application mode")
    input_values['Application_order'] = st.number_input("Application Order", min_value=0, value=1)
    input_values['Course'] = st.number_input("Course Code", min_value=0, value=1)
    input_values['Daytime_evening_attendance'] = st.selectbox("Daytime/Evening Attendance", options=[0, 1], format_func=lambda x: 'Daytime' if x == 1 else 'Evening', help="1 for Daytime, 0 for Evening")
    input_values['Previous_qualification'] = st.number_input("Previous Qualification", min_value=0, value=1, help="Code for previous qualification")
    input_values['Previous_qualification_grade'] = st.number_input("Previous Qualification Grade", value=100.0)
    input_values['Nacionality'] = st.number_input("Nationality", min_value=0, value=1, help="Code for nationality")

with col2:
    input_values['Mothers_qualification'] = st.number_input("Mother's Qualification", min_value=0, value=1, help="Code for mother's qualification")
    input_values['Fathers_qualification'] = st.number_input("Father's Qualification", min_value=0, value=1, help="Code for father's qualification")
    input_values['Mothers_occupation'] = st.number_input("Mother's Occupation", min_value=0, value=1, help="Code for mother's occupation")
    input_values['Fathers_occupation'] = st.number_input("Father's Occupation", min_value=0, value=1, help="Code for father's occupation")
    input_values['Admission_grade'] = st.number_input("Admission Grade", value=100.0)
    input_values['Displaced'] = st.selectbox("Displaced", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if displaced, 0 otherwise")
    input_values['Educational_special_needs'] = st.selectbox("Educational Special Needs", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if has special needs, 0 otherwise")
    input_values['Debtor'] = st.selectbox("Debtor", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if debtor, 0 otherwise")


col3, col4 = st.columns(2)
with col3:
    input_values['Tuition_fees_up_to_date'] = st.selectbox("Tuition Fees Up-to-Date", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if fees up-to-date, 0 otherwise")
    input_values['Gender'] = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Female' if x == 1 else 'Male', help="1 for Female, 0 for Male")
    input_values['Scholarship_holder'] = st.selectbox("Scholarship Holder", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if scholarship holder, 0 otherwise")
    input_values['Age_at_enrollment'] = st.number_input("Age at Enrollment", min_value=0, value=18)
    input_values['International'] = st.selectbox("International Student", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if international, 0 otherwise")

st.header("Academic Performance (1st Semester)")
col5, col6 = st.columns(2)
with col5:
    input_values['Curricular_units_1st_sem_credited'] = st.number_input("CU 1st Sem Credited", min_value=0, value=0)
    input_values['Curricular_units_1st_sem_enrolled'] = st.number_input("CU 1st Sem Enrolled", min_value=0, value=6)
    input_values['Curricular_units_1st_sem_evaluations'] = st.number_input("CU 1st Sem Evaluations", min_value=0, value=6)
with col6:
    input_values['Curricular_units_1st_sem_approved'] = st.number_input("CU 1st Sem Approved", min_value=0, value=6)
    input_values['Curricular_units_1st_sem_grade'] = st.number_input("CU 1st Sem Grade (avg)", value=14.0)
    input_values['Curricular_units_1st_sem_without_evaluations'] = st.number_input("CU 1st Sem Without Evaluations", min_value=0, value=0)

st.header("Academic Performance (2nd Semester)")
col7, col8 = st.columns(2)
with col7:
    input_values['Curricular_units_2nd_sem_credited'] = st.number_input("CU 2nd Sem Credited", min_value=0, value=0)
    input_values['Curricular_units_2nd_sem_enrolled'] = st.number_input("CU 2nd Sem Enrolled", min_value=0, value=6)
    input_values['Curricular_units_2nd_sem_evaluations'] = st.number_input("CU 2nd Sem Evaluations", min_value=0, value=6)
with col8:
    input_values['Curricular_units_2nd_sem_approved'] = st.number_input("CU 2nd Sem Approved", min_value=0, value=6)
    input_values['Curricular_units_2nd_sem_grade'] = st.number_input("CU 2nd Sem Grade (avg)", value=14.0)
    input_values['Curricular_units_2nd_sem_without_evaluations'] = st.number_input("CU 2nd Sem Without Evaluations", min_value=0, value=0)

st.header("Macroeconomic Factors")
col9, col10 = st.columns(2)
with col9:
    input_values['Unemployment_rate'] = st.number_input("Unemployment Rate", value=6.0)
with col10:
    input_values['Inflation_rate'] = st.number_input("Inflation Rate", value=2.0)
    input_values['GDP'] = st.number_input("GDP", value=1.0)


# --- Prediction ---
if st.button("Predict Status"):
    # Create a DataFrame from input values with the EXACT original column order
    input_data = pd.DataFrame([input_values], columns=original_feature_cols)

    # Make prediction using the pipeline
    prediction_numerical = pipeline.predict(input_data)

    # Decode the numerical prediction back to the original label
    prediction_label = le.inverse_transform(prediction_numerical)

    st.subheader("Prediction Result:")
    # Assuming 0 is Dropout and 1 is Graduate based on your previous confusion matrix
    # If your LabelEncoder mapped them differently, adjust the output message
    status_mapping = {
        le.transform(['Dropout'])[0]: 'Dropout', # Get the numerical code for 'Dropout'
        le.transform(['Graduate'])[0]: 'Graduate' # Get the numerical code for 'Graduate'
    }
    predicted_status_text = status_mapping.get(prediction_numerical[0], f"Unknown Status Code: {prediction_numerical[0]}")


    st.write(f"Predicted Status: **{predicted_status_text}**")

st.markdown("---")
st.write("Note: This is a predictive model and results should be interpreted with caution.")
st.write("Please ensure input values are accurate and reflect the same scale and meaning as the training data.")
