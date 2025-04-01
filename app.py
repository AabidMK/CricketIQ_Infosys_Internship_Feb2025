import streamlit as st
import joblib
import numpy as np

# Load trained models and encoders
model = joblib.load("final_rf_model.pkl")
team_encoder = joblib.load("le_team.pkl")
venue_encoder = joblib.load("le_venue.pkl")

# Streamlit UI
st.title("Cricket Match Winner🏆 Prediction")

# Dropdown for innings selection
innings = st.sidebar.selectbox("Select Innings", [1, 2])

# Dropdowns for categorical inputs
batting_team = st.sidebar.selectbox("Select Batting Team", team_encoder.classes_)
bowling_team = st.sidebar.selectbox("Select Bowling Team", team_encoder.classes_)
venue = st.sidebar.selectbox("Select Venue", venue_encoder.classes_)

# Numerical inputs with increase/decrease buttons
cum_runs = st.sidebar.number_input("Enter Cumulative Runs", min_value=0, step=1)
cum_wickets = st.sidebar.number_input("Enter Cumulative Wickets", min_value=0, max_value=10, step=1)
overs_completed = st.sidebar.number_input("Enter Overs Completed", min_value=0.0, max_value=50.0, step=0.1)

# Show Target Score input only for Innings 1
target = None
if innings == 2:
    target = st.sidebar.number_input("Enter Target Score", min_value=1, step=1)
else: 
    print("(N/A)")

# Calculate derived features
current_run_rate = cum_runs / overs_completed if overs_completed > 0 else 0.0

# Required run rate calculation (only for Innings 2, else set placeholder)
if innings == 2:
    remaining_overs = 50 - overs_completed
    required_run_rate = ((target - cum_runs) / remaining_overs) if remaining_overs > 0 else 0.0
else:
    required_run_rate = 0.0  # Placeholder for Innings 1

# Display calculated values
st.write(f"Current Run Rate: {current_run_rate:.2f}")
st.write(f"Required Run Rate: {required_run_rate:.2f}" )

# Predict Button
if st.button("🏆 Predict Winner"):
    # Encode categorical features
    batting_team_encoded = team_encoder.transform([batting_team])[0]
    bowling_team_encoded = team_encoder.transform([bowling_team])[0]
    venue_encoded = venue_encoder.transform([venue])[0]

    # Prepare input data
    input_data = np.array([[innings, batting_team_encoded, bowling_team_encoded, venue_encoded, cum_runs, cum_wickets, current_run_rate, required_run_rate, target if target else 0]])

    # Make prediction
    prediction = model.predict(input_data)

    if prediction == 1:
        winner = batting_team
    else:
        winner = bowling_team

    #  Display result
    st.success(f"Predicted Winner🏆: {winner}")

    probabilities = model.predict_proba(input_data)[0]
    batting_team_prob = probabilities[1] * 100
    bowling_team_prob = probabilities[0] * 100

    # Display result
    st.subheader("Winning Probabilities")
    st.write(f"{batting_team}: {batting_team_prob:.2f}%")
    st.write(f"{bowling_team}: {bowling_team_prob:.2f}%")

