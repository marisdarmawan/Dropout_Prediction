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

# --- Dictionaries for Categorical Mapping
marital_status_map = {
    'Single': 1,
    'Married': 2,
    'Widower': 3,
    'Divorced': 4,
    'Facto Union': 5,
    'Legally Separated': 6
}

application_mode_map = {
    '1st phase - general contingent': 1,
    'Ordinance No. 612/93': 2,
    '1st phase - special contingent (Azores Island)': 5,
    'Holders of other higher courses': 7,
    'Ordinance No. 854-B/99': 10,
    'International student (bachelor)': 15,
    '1st phase - special contingent (Madeira Island)': 16,
    '2nd phase - general contingent': 17,
    '3rd phase - general contingent': 18,
    'Ordinance No. 533-A/99, item b2) (Different Plan)': 26,
    'Ordinance No. 533-A/99, item b3 (Other Institution)': 27,
    'Over 23 years old': 39,
    'Transfer': 42,
    'Change of course': 43,
    'Technological specialization diploma holders': 44,
    'Change of institution/course': 51,
    'Short cycle diploma holders': 53,
    'Change of institution/course (International)': 57
}

course_map = {
    'Biofuel Production Technologies': 33,
    'Animation and Multimedia Design': 171,
    'Social Service (evening attendance)': 8014,
    'Agronomy': 9003,
    'Communication Design': 9070,
    'Veterinary Nursing': 9085,
    'Informatics Engineering': 9119,
    'Equinculture': 9130,
    'Management': 9147,
    'Social Service': 9238,
    'Tourism': 9254,
    'Nursing': 9500,
    'Oral Hygiene': 9556,
    'Advertising and Marketing Management': 9670,
    'Journalism and Communication': 9773,
    'Basic Education': 9853,
    'Management (evening attendance)': 9991
}

attendance_map = {'Daytime': 1, 'Evening': 0}

prev_qual_map = {
    'Secondary education': 1,
    "Higher education - bachelor's degree": 2,
    'Higher education - degree': 3,
    "Higher education - master's": 4,
    'Higher education - doctorate': 5,
    'Frequency of higher education': 6,
    '12th year of schooling - not completed': 9,
    '11th year of schooling - not completed': 10,
    'Other - 11th year of schooling': 12,
    '10th year of schooling': 14,
    '10th year of schooling - not completed': 15,
    'Basic education 3rd cycle (9th/10th/11th year) or equiv.': 19,
    'Basic education 2nd cycle (6th/7th/8th year) or equiv.': 38,
    'Technological specialization course': 39,
    'Higher education - degree (1st cycle)': 40,
    'Professional higher technical course': 42,
    'Higher education - master (2nd cycle)': 43
}

nationality_map = {
    'Portuguese': 1, 'German': 2, 'Spanish': 6, 'Italian': 11,
    'Dutch': 13, 'English': 14, 'Lithuanian': 17, 'Angolan': 21,
    'Cape Verdean': 22, 'Guinean': 24, 'Mozambican': 25, 'Santomean': 26,
    'Turkish': 32, 'Brazilian': 41, 'Romanian': 62, 'Moldova (Republic of)': 100,
    'Mexican': 101, 'Ukrainian': 103, 'Russian': 105, 'Cuban': 108,
    'Colombian': 109
}

parent_qual_map = {
    "Secondary Education - 12th Year of Schooling or Eq.": 1,
    "Higher Education - Bachelor's Degree": 2,
    "Higher Education - Degree": 3,
    "Higher Education - Master's": 4,
    "Higher Education - Doctorate": 5,
    "Frequency of Higher Education": 6,
    "12th Year of Schooling - Not Completed": 9,
    "11th Year of Schooling - Not Completed": 10,
    "7th Year (Old)": 11,
    "Other - 11th Year of Schooling": 12,
    "10th Year of Schooling": 14,
    "General commerce course": 18,
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19,
    "Technical-professional course": 22,
    "7th year of schooling": 26,
    "2nd cycle of the general high school course": 27,
    "9th Year of Schooling - Not Completed": 29,
    "8th year of schooling": 30,
    "Unknown": 34,
    "Can't read or write": 35,
    "Can read without having a 4th year of schooling": 36,
    "Basic education 1st cycle (4th/5th year) or equiv.": 37,
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Specialized higher studies course": 41,
    "Professional higher technical course": 42,
    "Higher Education - Master (2nd cycle)": 43,
    "Higher Education - Doctorate (3rd cycle)": 44,
    "2nd year complementary high school course": 13,
    "Complementary High School Course": 20,
    "Complementary High School Course - not concluded": 25,
    "General Course of Administration and Commerce": 31,
    "Supplementary Accounting and Administration": 33
}

parent_job_map = {
    "Student": 0,
    "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1,
    "Specialists in Intellectual and Scientific Activities": 2,
    "Intermediate Level Technicians and Professions": 3,
    "Administrative staff": 4,
    "Personal Services, Security and Safety Workers and Sellers": 5,
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6,
    "Skilled Workers in Industry, Construction and Craftsmen": 7,
    "Installation and Machine Operators and Assembly Workers": 8,
    "Unskilled Workers": 9,
    "Armed Forces Professions": 10,
    "Other Situation": 90,
    "(blank)": 99, 
    "Health professionals": 122,
    "Teachers": 123,
    "Specialists in information and communication technologies (ICT)": 125,
    "Intermediate level science and engineering technicians and professions": 131,
    "Technicians and professionals, of intermediate level of health": 132,
    "Intermediate level technicians from legal, social, sports, cultural and similar services": 134,
    "Office workers, secretaries in general and data processing operators": 141,
    "Data, accounting, statistical, financial services and registry-related operators": 143,
    "Other administrative support staff": 144,
    "Personal service workers": 151,
    "Sellers": 152,
    "Personal care workers and the like": 153,
    "Skilled construction workers and the like, except electricians": 171,
    "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like": 173,
    "Workers in food processing, woodworking, clothing and other industries and crafts": 175,
    "Cleaning workers": 191,
    "Unskilled workers in agriculture, animal production, fisheries and forestry": 192,
    "Unskilled workers in extractive industry, construction, manufacturing and transport": 193,
    "Meal preparation assistants": 194,
    "Armed Forces Officers": 101,
    "Armed Forces Sergeants": 102,
    "Other Armed Forces personnel": 103,
    "Directors of administrative and commercial services": 112,
    "Hotel, catering, trade and other services directors": 114,
    "Specialists in the physical sciences, mathematics, engineering and related techniques": 121,
    "Specialists in finance, accounting, administrative organization, public and commercial relations": 124,
    "Information and communication technology technicians": 135,
    "Protection and security services personnel": 154,
    "Market-oriented farmers and skilled agricultural and animal production workers": 161,
    "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence": 163,
    "Skilled workers in metallurgy, metalworking and similar": 172,
    "Skilled workers in electricity and electronics": 174,
    "Fixed plant and machine operators": 181,
    "Assembly workers": 182,
    "Vehicle drivers and mobile equipment operators": 183,
    "Street vendors (except food) and street service providers": 195
}

boolean_map = {'Yes': 1, 'No': 0}
gender_map = {'Male': 1, 'Female': 0}

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wider layout
st.title("🎓 Student Outcome Prediction")
st.write("""
This app predicts whether a student is likely to drop out, stay enrolled, or graduate
based on their academic and demographic information. Please input the student details on the sidebar and click predict.
""")

st.sidebar.header("Input Student Information")

col1, col2 = st.columns(2)

with col1:
    marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
    nacionality_desc = st.sidebar.selectbox("Nationality", list(nationality_map.keys()), index=0) # Default to Portuguese
    gender = st.sidebar.radio("Gender", list(gender_map.keys()))
    age = st.sidebar.slider("Age at Enrollment", 17, 70, 20) # Min age based on typical enrollment
    displaced = st.sidebar.radio("Is the student displaced?", list(boolean_map.keys()), index=1) # Default No
    special_needs = st.sidebar.radio("Educational Special Needs?", list(boolean_map.keys()), index=1) # Default No
    international = st.sidebar.radio("International Student?", list(boolean_map.keys()), index=1) # Default No

    application_mode = st.sidebar.selectbox("Application Mode", list(application_mode_map.keys()))
    application_order = st.sidebar.slider("Application Order (0 = 1st choice)", 0, 9, 0)
    course = st.sidebar.selectbox("Course", list(course_map.keys()))
    attendance = st.sidebar.radio("Class Attendance", list(attendance_map.keys()))
    admission_grade = st.sidebar.slider("Admission Grade", 0, 200, 140) # Default 140
    prev_qualification_desc = st.sidebar.selectbox("Previous Qualification", list(prev_qual_map.keys()), index=0) # Default Secondary
    prev_grade = st.sidebar.slider("Previous Qualification Grade", 0, 200, 140) # Default 140

with col2:
    mother_qual_desc = st.sidebar.selectbox("Mother's Qualification", list(parent_qual_map.keys()), index=0)
    father_qual_desc = st.sidebar.selectbox("Father's Qualification", list(parent_qual_map.keys()), index=0)
    mother_job_desc = st.sidebar.selectbox("Mother's Occupation", list(parent_job_map.keys()), index=11) # Default to (blank) or common category
    father_job_desc = st.sidebar.selectbox("Father's Occupation", list(parent_job_map.keys()), index=11) # Default to (blank) or common category

    debtor = st.sidebar.radio("Is the student a debtor?", list(boolean_map.keys()), index=1) # Default No
    fees_up_to_date = st.sidebar.radio("Tuition Fees Up To Date?", list(boolean_map.keys()), index=0) # Default Yes
    scholarship = st.sidebar.radio("Scholarship Holder?", list(boolean_map.keys()), index=1) # Default No

    cred_1st = st.sidebar.slider("1st Sem: Credited Units", 0, 30, 0) 
    enr_1st = st.sidebar.slider("1st Sem: Enrolled Units", 0, 30, 6)   
    eval_1st = st.sidebar.slider("1st Sem: Evaluated Units", 0, 30, 6) 
    appr_1st = st.sidebar.slider("1st Sem: Approved Units", 0, 30, 5) 
    grade_1st = st.sidebar.slider("1st Sem: Average Grade", 0.0, 20.0, 12.0, step=0.1) 
    no_eval_1st = st.sidebar.slider("1st Sem: Units Without Evaluation", 0, 30, 0)

    cred_2nd = st.sidebar.slider("2nd Sem: Credited Units", 0, 30, 0) 
    enr_2nd = st.sidebar.slider("2nd Sem: Enrolled Units", 0, 30, 6)   
    eval_2nd = st.sidebar.slider("2nd Sem: Evaluated Units", 0, 30, 6) 
    appr_2nd = st.sidebar.slider("2nd Sem: Approved Units", 0, 30, 5)  
    grade_2nd = st.sidebar.slider("2nd Sem: Average Grade", 0.0, 20.0, 12.0, step=0.1)
    no_eval_2nd = st.sidebar.slider("2nd Sem: Units Without Evaluation", 0, 30, 0) 

    unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 20.0, 10.0, step=0.1)
    inflation_rate = st.sidebar.slider("Inflation Rate (%)", -5.0, 10.0, 1.5, step=0.1)
    gdp = st.sidebar.slider("GDP Growth Rate (%)", -10.0, 10.0, 0.5, step=0.1)

# Map selected descriptions back to codes
nationality_code = nationality_map[nacionality_desc]
prev_qualification_code = prev_qual_map[prev_qualification_desc]
mother_qual_code = parent_qual_map[mother_qual_desc]
father_qual_code = parent_qual_map[father_qual_desc]
mother_job_code = parent_job_map[mother_job_desc]
father_job_code = parent_job_map[father_job_desc]


input_data_list = [
    marital_status_map[marital_status],
    application_mode_map[application_mode],
    application_order,
    course_map[course],
    attendance_map[attendance],
    prev_qualification_code, 
    prev_grade,
    nationality_code,       
    mother_qual_code,        
    father_qual_code,        
    mother_job_code,         
    father_job_code,         
    admission_grade,
    boolean_map[displaced],
    boolean_map[special_needs],
    boolean_map[debtor],
    boolean_map[fees_up_to_date],
    gender_map[gender],
    boolean_map[scholarship],
    age,
    boolean_map[international],
    cred_1st,
    enr_1st,
    eval_1st,
    appr_1st,
    grade_1st,               
    no_eval_1st,            
    cred_2nd,               
    enr_2nd,                 
    eval_2nd,                
    appr_2nd,                
    grade_2nd,               
    no_eval_2nd,             
    unemployment_rate,       
    inflation_rate,          
    gdp                      
]

# Convert to NumPy array with the correct shape (1 row, N columns)
input_data = np.array([input_data_list])

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

# --- Prediction ---
if st.button("Predict Status"):
    # Create a DataFrame from input values with the EXACT original column order
    input_data = pd.DataFrame([input_data], columns=original_feature_cols)

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
