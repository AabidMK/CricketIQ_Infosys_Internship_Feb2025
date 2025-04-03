import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Encoders and Model
le_team = joblib.load('team_encoder.pkl')
le_venue = joblib.load('venue_encoder.pkl')
model = joblib.load('final_rf_model.pkl')

# Set Page Config
st.set_page_config(page_title="IPL Match Outcome Prediction", page_icon="🏏", layout="wide")

# Title with Icon
st.markdown(
    "<h1 style='text-align: center; color: #ff4b4b;'>🏏 IPL Match Outcome Prediction 🏆</h1>",
    unsafe_allow_html=True,
)

# Sidebar for Inputs
st.sidebar.header("📊 Match Details")
inning = st.sidebar.selectbox("Innings", [1, 2])
batting_team = st.sidebar.selectbox("Batting Team", le_team.classes_)
bowling_team = st.sidebar.selectbox("Bowling Team", le_team.classes_)
venue = st.sidebar.selectbox("Venue", le_venue.classes_)

total_runs = st.sidebar.number_input("Total Runs", min_value=0, max_value=300, step=1)
wickets = st.sidebar.number_input("Wickets", min_value=0, max_value=10, step=1)
overs_completed = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)

# Target score only applicable if 2nd inning
target_score = st.sidebar.number_input("Target Score", min_value=0, max_value=300, step=1) if inning == 2 else 0

# Columns for Run Rate
col1, col2 = st.columns(2)

# Calculate Current Run Rate
current_run_rate = total_runs / overs_completed if overs_completed > 0 else 0
col1.metric("📈 Current Run Rate", f"{current_run_rate:.2f}", delta_color="off")

# Default value for required_run_rate
required_run_rate = 0

# Calculate Required Run Rate only in 2nd inning
if inning == 2 and target_score > 0 and total_runs < target_score:
    remaining_runs = target_score - total_runs
    remaining_overs = 20 - overs_completed
    required_run_rate = remaining_runs / remaining_overs if remaining_overs > 0 else 0
    col2.metric("📉 Required Run Rate", f"{required_run_rate:.2f}", delta_color="inverse" if required_run_rate > 10 else "normal")
else:
    col2.metric("📉 Required Run Rate", "N/A")

# Encode Inputs
batting_team_encoded = le_team.transform([batting_team])[0]
bowling_team_encoded = le_team.transform([bowling_team])[0]
venue_encoded = le_venue.transform([venue])[0]

# Prepare Input Data
input_data = pd.DataFrame({
    'inning': [inning],
    'cum_runs': [total_runs],
    'cum_wickets': [wickets],
    'current_run_rate': [current_run_rate],
    'required_run_rate': [required_run_rate],
    'target_score': [target_score],
    'batting_team_encoded': [batting_team_encoded],
    'bowling_team_encoded': [bowling_team_encoded],
    'venue_canonical_encoded': [venue_encoded]
})

# Button to Predict Winner
if st.button("🔮 Predict Outcome"):
    prediction = model.predict(input_data)[0]
    predicted_winner = batting_team if prediction == 1 else bowling_team
    st.success(f"🏆 Predicted Winner: {predicted_winner}")


