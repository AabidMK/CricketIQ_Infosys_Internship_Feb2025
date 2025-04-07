import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load Files
model = joblib.load("C:/Users/Dell/Desktop/IPL Visualization/random_forest_model.pkl")
team_encoder = joblib.load("C:/Users/Dell/Desktop/IPL Visualization/team_encoder.pkl")
venue_encoder = joblib.load("C:/Users/Dell/Desktop/IPL Visualization/venue_encoder.pkl")

# List Of Teams And Venues
teams_list = team_encoder.classes_.tolist()
venues_list = venue_encoder.classes_.tolist()

print(teams_list)
print(venues_list)

# Title
st.title("Match Winning Prediction 🏏")

# Sidebar
st.sidebar.header("Input Match Details")

# DropDown Inputs 
inning = st.sidebar.selectbox("Innings", (1, 2))

# Dropdown for Batting Teams
batting_team = st.sidebar.selectbox("Batting Team", teams_list)

# Filter bowling team options (excluding selected batting team)
bowling_options = [team for team in teams_list if team != batting_team]
bowling_team = st.sidebar.selectbox("Bowling Team", bowling_options)

# Venue Dropdown
venue_name = st.sidebar.selectbox("Venue", venues_list)

# Inputs
cumulative_runs = st.sidebar.number_input("Cumulative Runs", min_value=0, value=70, step=1)
cumulative_wickets = st.sidebar.number_input("Cumulative Wickets", min_value=0, value=2, step=1)
overs_completed = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=7.0, step=0.1)

# If Inning = 2, allow user to enter target score
target = st.sidebar.number_input("Target Score", min_value=0, value=160, step=1) if inning == 2 else 0

# Derived Features
current_run_rate = cumulative_runs / overs_completed if overs_completed > 0 else 0
remaining_overs = 20 - overs_completed
required_run_rate = (target - cumulative_runs) / remaining_overs if inning == 2 and remaining_overs > 0 else 0

#Display Features 
st.markdown("Derived Features🚀")
st.write(f"Current Run Rate : {current_run_rate:.2f}")
st.write(f"Required Run Rate : {required_run_rate:.2f}")

# Encode Categorical Values
batting_team_encoded = team_encoder.transform([batting_team])[0]
bowling_team_encoded = team_encoder.transform([bowling_team])[0]
venue_encoded = venue_encoder.transform([venue_name])[0]

# Input DataFrame
input_data = pd.DataFrame({
    'inning': [inning],
    'cumulative_runs': [cumulative_runs],
    'cumulative_wickets': [cumulative_wickets],
    'current_run_rate': [current_run_rate],
    'target': [target],
    'required_run_rate': [required_run_rate],
    'batting_team_encoded': [batting_team_encoded],
    'bowling_team_encoded': [bowling_team_encoded],
    'venue_encoded': [venue_encoded]
})

# Prediction
prediction = model.predict(input_data)[0]
prediction_probability = model.predict_proba(input_data)[0]

# winner prediction
predicted_winner = batting_team if prediction == 1 else bowling_team

# Display prediction
st.write("Winner Prediction🤖 ")
st.write(f"{batting_team} winning probability : {prediction_probability[1] * 100:.2f}%")
st.write(f"{bowling_team} winning probability : {prediction_probability[0] * 100:.2f}%")
#st.write("🏆 Predicted Winner: ", predicted_winner)

#Display Styling
st.markdown(
    f"""
    <div style="
        padding: 20px; 
        #background-color:gold;
        background: linear-gradient(to left, gold,orange);
        color: black;  
        font-size: 25px;
        font-weight: bold;
        box-shadow: 0px 3px 8px white;
        text-align: center;
        border-radius: 12px;
    ">
        🏆   Predicted Winner: {predicted_winner}
    </div>
    """, 
    unsafe_allow_html=True
)