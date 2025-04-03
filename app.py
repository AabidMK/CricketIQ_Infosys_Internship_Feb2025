import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and encoders
model = joblib.load('best_model.pkl')
encoders = joblib.load('encoders.pkl')

st.title('IPL Match Outcome Predictor')
st.write("Enter the match details to predict the outcome.")

# User Inputs
batting_team = st.selectbox('Select Batting Team', encoders['batting_team'].classes_)
bowling_team = st.selectbox('Select Bowling Team', encoders['bowling_team'].classes_)
venue = st.selectbox('Select Venue', encoders['venue'].classes_)
cum_runs = st.number_input('Cumulative Runs', min_value=0)
cum_wickets = st.number_input('Cumulative Wickets', min_value=0, max_value=10)
current_run_rate = st.number_input('Current Run Rate', min_value=0.0)
required_run_rate = st.number_input('Required Run Rate', min_value=0.0)
target = st.number_input('Target Score', min_value=1)

# Data Preprocessing
input_data = pd.DataFrame({
    'cum_runs': [cum_runs],
    'cum_wickets': [cum_wickets],
    'current_run_rate': [current_run_rate],
    'required_run_rate': [required_run_rate],
    'target': [target],
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'venue': [venue]
})

# Encode Categorical Variables
for col in ['batting_team', 'bowling_team', 'venue']:
    input_data[col] = encoders[col].transform(input_data[col])

# Predict Outcome
if st.button('Predict Outcome'):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success('Prediction: The batting team is predicted to WIN!')
    else:
        st.error('Prediction: The batting team is predicted to LOSE!')
