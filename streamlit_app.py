import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(r"C:\Users\Giriraj S A\Downloads\final_rf_model.pkl")
le_team = joblib.load(r"C:\Users\Giriraj S A\Downloads\le_team.pkl")
le_venue = joblib.load(r"C:\Users\Giriraj S A\Downloads\le_venue.pkl")

# Streamlit UI
st.title("Cricket Match Winner Prediction")

st.write("""
    Enter the match details below to predict the outcome (Win/Loss).
""")

# Input fields for match details
innings = st.selectbox("Innings", [1, 2])
batting_team = st.selectbox("Batting Team", le_team.classes_)
bowling_team = st.selectbox("Bowling Team", le_team.classes_)
venue = st.selectbox("Venue", le_venue.classes_)
total_runs = st.number_input("Total Runs", min_value=0, step=1)
cumulative_wickets = st.number_input("Cumulative Wickets", min_value=0, step=1)
overs_completed = st.number_input("Overs Completed", min_value=0.0, step=0.1)
target_runs = st.number_input("Target Runs", min_value=0, step=1)

# Calculate wicket impact
wicket_impact = cumulative_wickets / 10

# Prepare the input data for prediction
if st.button("Predict Match Outcome"):
    # Encode team and venue names
    batting_team_encoded = le_team.transform([batting_team])[0]
    bowling_team_encoded = le_team.transform([bowling_team])[0]
    venue_encoded = le_venue.transform([venue])[0]

    # Prepare the input data as a DataFrame
    new_data = pd.DataFrame({
        'innings': [innings],
        'batting_team_encoded': [batting_team_encoded],
        'bowling_team_encoded': [bowling_team_encoded],
        'venue_encoded': [venue_encoded],
        'total_runs': [total_runs],
        'cumulative_wickets': [cumulative_wickets],
        'overs_completed': [overs_completed],
        'target_runs': [target_runs],
        'wicket_impact': [wicket_impact]
    })

    # Make prediction
    prediction = model.predict(new_data)

    # Display the result
    if prediction[0] == 1:
        st.write("Predicted Outcome: **Win**")
    else:
        st.write("Predicted Outcome: **Loss**")
