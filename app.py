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
        X['parental_education_avg'] = (X['Fathers_qualification'] + X['Mothers_qualification']) / 2 # Corrected typo Mothers_
        
        return X

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        if self.columns is None:
            # If columns are not specified, encode all object/category columns
            self.columns = X.select_dtypes(include=['object', 'category']).columns

        for col in self.columns:
            le = LabelEncoder()
            # Ensure all possible categories from training data are fit
            # This mapping should ideally come from the training phase for consistency
            le.fit(X[col].astype(str)) # Convert to string to handle mixed types
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                # Use a lambda function with apply to handle unseen labels gracefully
                X_transformed[col] = X_transformed[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        return X_transformed

# --- Load the trained pipeline ---
# Make sure 'student_status_pipeline.pkl' is in the same directory as this app.py
try:
    pipeline = joblib.load('student_status_pipeline.pkl')
except FileNotFoundError:
    st.error("Error: 'student_status_pipeline.pkl' not found. Please ensure the trained model pipeline is in the same directory.")
    st.stop()

# --- Define feature columns and prediction map ---
# These must exactly match the features used during the model training
original_feature_cols = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
    'Inflation_rate', 'GDP'
]

# Mapping prediction output to label
prediction_map = {
    0: "❌ The student is predicted to **DROP OUT**.",
    1: "🎓 The student is predicted to **GRADUATE**."    
}

# --- Streamlit App Layout ---
st.set_page_config(page_title="Student Status Prediction", layout="wide")
st.title("Student Status Prediction App")

st.sidebar.header("Navigation")
prediction_mode = st.sidebar.radio(
    "Choose Prediction Mode",
    ("Manual Input", "Upload CSV File")
)

if prediction_mode == "Manual Input":
    st.header("Predict Student Status - Manual Input")

    # --- Manual Input Fields ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Demographics")
        marital_status = st.number_input("Marital Status", min_value=1, max_value=6, value=1, help="1=Single, 2=Married, 3=Widower, etc.")
        application_mode = st.number_input("Application Mode", min_value=1, max_value=62, value=1, help="Mode of application")
        application_order = st.number_input("Application Order", min_value=0, max_value=8, value=1)
        course = st.number_input("Course ID", min_value=1, max_value=9999, value=9238)
        daytime_evening_attendance = st.selectbox("Daytime/Evening Attendance", [0, 1], format_func=lambda x: "Daytime" if x==1 else "Evening")
        previous_qualification = st.number_input("Previous Qualification", min_value=1, max_value=43, value=1)
        previous_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=130.0)
        nacionality = st.number_input("Nationality", min_value=1, max_value=101, value=1)
        mothers_qualification = st.number_input("Mother's Qualification", min_value=1, max_value=44, value=19)
        fathers_qualification = st.number_input("Father's Qualification", min_value=1, max_value=44, value=19)
        mothers_occupation = st.number_input("Mother's Occupation", min_value=0, max_value=145, value=9)
        fathers_occupation = st.number_input("Father's Occupation", min_value=0, max_value=145, value=9)
        admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=130.0)
        displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        educational_special_needs = st.selectbox("Educational Special Needs", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        tuition_fees_up_to_date = st.selectbox("Tuition Fees Up-to-Date", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        scholarship_holder = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        age_at_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20)
        international = st.selectbox("International Student", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with col2:
        st.subheader("Academic Performance & Environment")
        curricular_units_1st_sem_credited = st.number_input("1st Sem Credited Units", min_value=0, value=0)
        curricular_units_1st_sem_enrolled = st.number_input("1st Sem Enrolled Units", min_value=0, value=6)
        curricular_units_1st_sem_evaluations = st.number_input("1st Sem Evaluations", min_value=0, value=6)
        curricular_units_1st_sem_approved = st.number_input("1st Sem Approved Units", min_value=0, value=6)
        curricular_units_1st_sem_grade = st.number_input("1st Sem Grade", min_value=0.0, value=13.0)
        curricular_units_1st_sem_without_evaluations = st.number_input("1st Sem Without Evaluations", min_value=0, value=0)

        curricular_units_2nd_sem_credited = st.number_input("2nd Sem Credited Units", min_value=0, value=0)
        curricular_units_2nd_sem_enrolled = st.number_input("2nd Sem Enrolled Units", min_value=0, value=6)
        curricular_units_2nd_sem_evaluations = st.number_input("2nd Sem Evaluations", min_value=0, value=6)
        curricular_units_2nd_sem_approved = st.number_input("2nd Sem Approved Units", min_value=0, value=6)
        curricular_units_2nd_sem_grade = st.number_input("2nd Sem Grade", min_value=0.0, value=13.0)
        curricular_units_2nd_sem_without_evaluations = st.number_input("2nd Sem Without Evaluations", min_value=0, value=0)

        unemployment_rate = st.number_input("Unemployment Rate (at enrollment)", min_value=0.0, value=10.0)
        inflation_rate = st.number_input("Inflation Rate (at enrollment)", min_value=-5.0, value=1.0)
        gdp = st.number_input("GDP (at enrollment)", min_value=-10.0, value=2.0)

    # Collect inputs into a list matching the original_feature_cols order
    input_data_list = [
        marital_status, application_mode, application_order, course,
        daytime_evening_attendance, previous_qualification,
        previous_qualification_grade, nacionality, mothers_qualification,
        fathers_qualification, mothers_occupation, fathers_occupation,
        admission_grade, displaced, educational_special_needs, debtor,
        tuition_fees_up_to_date, gender, scholarship_holder,
        age_at_enrollment, international, curricular_units_1st_sem_credited,
        curricular_units_1st_sem_enrolled, curricular_units_1st_sem_evaluations,
        curricular_units_1st_sem_approved, curricular_units_1st_sem_grade,
        curricular_units_1st_sem_without_evaluations, curricular_units_2nd_sem_credited,
        curricular_units_2nd_sem_enrolled, curricular_units_2nd_sem_evaluations,
        curricular_units_2nd_sem_approved, curricular_units_2nd_sem_grade,
        curricular_units_2nd_sem_without_evaluations, unemployment_rate,
        inflation_rate, gdp
    ]

    # --- Prediction ---
    if st.button("Predict Status"):
        # Create a DataFrame from input values with the EXACT original column order
        input_features_np = np.array([input_data_list])
        input_df = pd.DataFrame(input_features_np, columns=original_feature_cols)
        
        # Make prediction using the pipeline
        prediction_numerical = pipeline.predict(input_df)[0] # Get single prediction

        # Predict probability
        prediction_proba = pipeline.predict_proba(input_df)[0]

        # Decode the numerical prediction back to the original label
        prediction_label = prediction_map.get(int(prediction_numerical), "Unknown Prediction") # Handle potential unexpected prediction values

        st.subheader("Prediction Result:")
        st.info(prediction_label)

        st.subheader("Prediction Probabilities")
        # Create a dataframe for better display of probabilities
        proba_df = pd.DataFrame({
            'Outcome': [prediction_map.get(i, f"Class {i}") for i in pipeline.classes_],
            'Probability': prediction_proba
        })
        st.dataframe(proba_df.style.format({'Probability': "{:.2%}"}), hide_index=True)


else: # prediction_mode == "Upload CSV File"
    st.header("Predict Student Status - Upload CSV File")
    st.write("Upload a CSV file containing student data for batch prediction.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file, assuming semicolon delimiter as per data - Copy.csv metadata
            df_csv = pd.read_csv(uploaded_file, delimiter=';') #
            st.success("CSV file uploaded successfully!")
            st.dataframe(df_csv.head())

            # --- Column Handling for uploaded CSV ---
            # Identify columns in CSV that are not in your model's expected features
            columns_to_drop = [col for col in df_csv.columns if col not in original_feature_cols]

            # Drop these columns from the DataFrame
            if columns_to_drop:
                st.warning(f"Dropping columns from CSV that are not model features: {', '.join(columns_to_drop)}")
                df_csv = df_csv.drop(columns=columns_to_drop)
            
            # Check if all required original_feature_cols are now in df_csv
            missing_features_after_drop = [f for f in original_feature_cols if f not in df_csv.columns]
            if missing_features_after_drop:
                st.error(f"Error: The uploaded CSV is missing expected model features: {', '.join(missing_features_after_drop)}. Please ensure your CSV contains all the necessary columns.")
            elif df_csv.empty:
                st.warning("The uploaded CSV file is empty after processing.")
            else:
                # Reorder columns to match the training data's feature order
                df_to_predict = df_csv[original_feature_cols]

                if st.button("Predict All Students"):
                    with st.spinner("Making predictions..."):
                        # Make predictions using the pipeline
                        batch_predictions_numerical = pipeline.predict(df_to_predict)
                        batch_probabilities = pipeline.predict_proba(df_to_predict)[:, 1] # Probability of graduating (class 1)

                        # Create a DataFrame for results
                        df_results = df_csv.copy() # Start with the original (cleaned) df_csv
                        df_results['Predicted_Status'] = [prediction_map.get(int(p), "Unknown") for p in batch_predictions_numerical]
                        df_results['Probability_Graduate'] = batch_probabilities
                        df_results['Probability_DropOut'] = 1 - batch_probabilities

                        st.subheader("Batch Prediction Results:")
                        st.dataframe(df_results)

                        # Provide download button
                        csv_output = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv_output,
                            file_name="student_predictions.csv",
                            mime="text/csv"
                        )
                        st.success("Predictions complete!")

        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}. Please ensure it's a valid CSV with the correct delimiter (;).")
