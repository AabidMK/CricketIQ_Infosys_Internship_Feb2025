import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ---------- Load Model & Encoders Safely ----------
@st.cache_resource
def load_model():
    model_path = "final_rf_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}. Please check the file path.")
        return None
    with open(model_path, "rb") as f:
        return joblib.load(f)

@st.cache_resource
def load_encoders():
    try:
        with open("le_team.pkl", "rb") as f:
            le_team = joblib.load(f)
        with open("le_venue.pkl", "rb") as f:
            le_venue = joblib.load(f)
        return le_team, le_venue
    except FileNotFoundError:
        st.error("⚠️ Label encoder files not found! Please check file paths.")
        st.stop()

# Load Model & Encoders
model = load_model()
if model is None:
    st.stop()

le_team, le_venue = load_encoders()

# Verify model input size
st.write("Model expects features:", model.n_features_in_)

# ---------- Streamlit UI ----------
st.title("🏏 IPL Match Win Predictor")
st.sidebar.header("Enter Match Details")

# Dropdown Inputs
inning = st.sidebar.radio("Inning", [1, 2])
batting_team = st.sidebar.selectbox("Batting Team", le_team.classes_)
bowling_team = st.sidebar.selectbox("Bowling Team", le_team.classes_)
venue = st.sidebar.selectbox("Venue", le_venue.classes_)

# Numeric Inputs
total_runs = st.sidebar.number_input("Total Runs", min_value=0, value=100)
cum_wickets = st.sidebar.slider("Wickets Lost", min_value=0, max_value=10, value=3)
overs_completed = st.sidebar.slider("Overs Completed", min_value=0.0, max_value=20.0, step=0.1, value=10.0)
target = st.sidebar.number_input("Target (only if 2nd innings)", min_value=0, value=200 if inning == 2 else 0)

# ---------- Feature Engineering ----------
remaining_runs = target - total_runs if inning == 2 else 0
remaining_overs = max(20 - overs_completed, 1)  # Avoid zero division
rrr = remaining_runs / remaining_overs if inning == 2 else 0  # Required Run Rate
crr = total_runs / overs_completed if overs_completed > 0 else 0  # Current Run Rate
wicket_impact = cum_wickets / 10  # Normalize wickets

st.sidebar.markdown(f"**Current Run Rate:** {crr:.2f}")
st.sidebar.markdown(f"**Required Run Rate:** {rrr:.2f}" if inning == 2 else "")

# Encode Teams & Venue
try:
    batting_team_encoded = le_team.transform([batting_team])[0]
    bowling_team_encoded = le_team.transform([bowling_team])[0]
    venue_encoded = le_venue.transform([venue])[0]
except ValueError as e:
    st.error(f"⚠️ Encoding error: {e}. Check if the selected team/venue exists in the dataset.")
    st.stop()

# Model Input (Fixed to 9 Features)
features = np.array([[inning, batting_team_encoded, bowling_team_encoded, venue_encoded, total_runs, cum_wickets, overs_completed, target, wicket_impact]])

# ---------- Prediction ----------
st.sidebar.markdown("### 🎯 Click below to predict!")
if st.sidebar.button("📊 Predict Outcome"):
    win_probability = model.predict_proba(features)[0][1] * 100
    lose_probability = 100 - win_probability
    winning_team = batting_team if win_probability > 50 else bowling_team

    st.subheader(f"🏆 {winning_team} is more likely to win!")
    st.metric(label="Win Probability", value=f"{win_probability:.2f}%")
    st.metric(label="Lose Probability", value=f"{lose_probability:.2f}%")
    
        
    # Visualize Probability using Plotly Stacked Bar Chart
    df = pd.DataFrame({
        "Team": [batting_team, bowling_team],
        "Win Probability": [win_probability, 0],
        "Lose Probability": [0, lose_probability]
    })
    fig_stacked_bar = px.bar(df, x="Team", y=["Win Probability", "Lose Probability"], 
                             title="Winning Probability", barmode="stack", 
                             labels={"value": "Probability (%)", "variable": "Outcome"},
                             color_discrete_map={"Win Probability": "green", "Lose Probability": "red"})
    st.plotly_chart(fig_stacked_bar)
    

