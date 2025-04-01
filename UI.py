import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model and encoders
model = joblib.load("final_rf_model.pkl")
encoders = joblib.load("encoders.pkl")  

# Extract LabelEncoders for teams and venues
team_encoder = encoders["batting_team"]
venue_encoder = encoders["venue_canonical"]

# Get the team and venue names from the encoder
team_names = team_encoder.classes_
venue_names = venue_encoder.classes_

# Streamlit UI
st.title(" Cricket Match Win Predictor")
st.sidebar.header("Enter Match Details")

# User Input Fields
inning = st.sidebar.selectbox("Inning", [1, 2])
cum_runs = st.sidebar.number_input("Cumulative Runs", min_value=0, value=80)
cum_wickets = st.sidebar.number_input("Cumulative Wickets", min_value=0, max_value=10, value=5)
overs_completed = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=12.0)

# Show "Target Score" only if Inning is 2
target = None  # Default when Inning = 1
if inning == 2:
    target = st.sidebar.number_input("Target Score", min_value=0, value=150)

# Use loaded team and venue names
batting_team = st.sidebar.selectbox("Batting Team", team_names)
bowling_team = st.sidebar.selectbox("Bowling Team", team_names)
venue = st.sidebar.selectbox("Venue", venue_names)

# Encode Categorical Data using the LabelEncoders
batting_team_encoded = team_encoder.transform([batting_team])[0]
bowling_team_encoded = team_encoder.transform([bowling_team])[0]
venue_encoded = venue_encoder.transform([venue])[0]

# Compute Derived Features
remaining_overs = 20 - overs_completed
current_run_rate = cum_runs / (overs_completed if overs_completed > 0 else 1)

# Only compute "Required Run Rate" if Inning = 2
required_run_rate = 0
if inning == 2 and target is not None:
    required_run_rate = (target - cum_runs) / (remaining_overs if remaining_overs > 0 else 1)

# Display Run Rates
st.markdown(" Match Stats")
st.write(f"**Current Run Rate:** {current_run_rate:.2f}")
if inning == 2:
    st.write(f"**Required Run Rate:** {required_run_rate:.2f}")

# Make Prediction
if st.sidebar.button("Predict Winner"):
    input_data = np.array([[
        inning, cum_runs, cum_wickets, current_run_rate, required_run_rate,
        (target if inning == 2 else 0),  # Ensure target is 0 for Inning 1
        batting_team_encoded, bowling_team_encoded, venue_encoded
    ]])
    prediction = model.predict(input_data)[0]
    
    # Display the actual team name instead of "Batting/Bowling Team Wins"
    result = f"{batting_team} Win!" if prediction == 1 else f"{bowling_team} Win!"
    st.subheader(f"Prediction: {result}")
