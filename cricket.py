import streamlit as st
import joblib
import pandas as pd

# Load the trained model and label encoders
model = joblib.load("final_rf_model.pkl")
laen_team = joblib.load("laen_team.pkl")
laen_venue = joblib.load("laen_venue.pkl")

# Streamlit UI
st.title("Cricket Match Winner Predictor 🏏")

# User inputs
inning = st.selectbox("Select Inning", [1, 2])
cum_runs = st.number_input("Cumulative Runs", min_value=0)
cum_wickets = st.number_input("Cumulative Wickets", min_value=0, max_value=10)
overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
target = st.number_input("Target Score (Enter 0 for 1st Inning)", min_value=0)

batting_team = st.text_input("Batting Team")
bowling_team = st.text_input("Bowling Team")
venue = st.text_input("Match Venue")

if st.button("Predict Winner"):
    # Calculate run rates
    current_run_rate = cum_runs / overs_completed if overs_completed > 0 else 0
    remaining_overs = 20 - overs_completed
    required_run_rate = (target - cum_runs) / remaining_overs if inning == 2 and remaining_overs > 0 else 0

    # Encode categorical values
    batting_team_encoded = laen_team.transform([batting_team])[0]
    bowling_team_encoded = laen_team.transform([bowling_team])[0]
    venue_encoded = laen_venue.transform([venue])[0]

    # Create input dataframe
    input_data = pd.DataFrame({
        'inning': [inning],
        'cum_runs': [cum_runs],
        'cum_wickets': [cum_wickets],
        'current_run_rate': [current_run_rate],
        'required_run_rate': [required_run_rate],
        'target': [target],
        'batting_team_encoded': [batting_team_encoded],
        'bowling_team_encoded': [bowling_team_encoded],
        'venue_canonical_encoded': [venue_encoded]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    predicted_probabilities = model.predict_proba(input_data)[0]

    # Display results
    predicted_winner = batting_team if prediction == 1 else bowling_team
    st.success(f"Predicted Winner: {predicted_winner}")
    st.write(f"Win Probability: {predicted_probabilities[1] * 100:.2f}%")
