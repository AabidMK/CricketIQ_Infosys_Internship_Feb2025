import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pyngrok import ngrok

le_team = joblib.load('le_team123.pkl')
le_venue = joblib.load('le_venue123.pkl')
model = joblib.load('final_rf_model.pkl')

# Streamlit Page Config
st.set_page_config(layout="wide", page_title="🏏 Cricket Match Predictor", page_icon="🏆")

# Title
st.markdown("🏏 Cricket Match Predictor", unsafe_allow_html=True)

# Sidebar for Input Features
st.sidebar.header("🔢 Match Input Features")
inning = st.sidebar.number_input("📌 Inning", min_value=1, max_value=2, value=1)
cum_runs = st.sidebar.number_input("🏏 Cumulative Runs", min_value=0, value=80)
cum_wickets = st.sidebar.number_input("❌ Cumulative Wickets", min_value=0, value=5)
overs_completed = st.sidebar.number_input("⏳ Overs Completed", min_value=0.0, value=12.0)
target = st.sidebar.number_input("🎯 Target Score", min_value=0, value=150)
batting_team = st.sidebar.selectbox("🏏 Batting Team", le_team.classes_)
bowling_team = st.sidebar.selectbox("🎯 Bowling Team", le_team.classes_)
venue = st.sidebar.selectbox("📍 Venue", le_venue.classes_)

# Prediction Button
if st.sidebar.button("🚀 Predict Match Outcome"):
    # Feature Engineering
    current_run_rate = cum_runs / max(overs_completed, 1)
    remaining_overs = max(20 - overs_completed, 1)
    required_run_rate = (target - cum_runs) / remaining_overs

    batting_team_encoded = le_team.transform([batting_team])[0]
    bowling_team_encoded = le_team.transform([bowling_team])[0]
    venue_encoded = le_venue.transform([venue])[0]

    input_df = pd.DataFrame([{
        "inning": inning,
        "cum_runs": cum_runs,
        "cum_wickets": cum_wickets,
        "overs_completed": overs_completed,
        "target": target,
        "batting_team": batting_team_encoded,
        "bowling_team": bowling_team_encoded,
        "venue_encoded": venue_encoded,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate
    }])

    input_df = input_df[model.feature_names_in_]
    predicted_probabilities = model.predict_proba(input_df)[0]

    # Determine Winner
    teams = [batting_team, bowling_team]
    winner_index = np.argmax(predicted_probabilities)
    predicted_winner = teams[winner_index]
    predicted_loser = teams[1 - winner_index]

    # Display Result
    st.markdown(f"🎉 **Predicted Winner:** {predicted_winner} 🏆", unsafe_allow_html=True)

    # Donut Chart for Probability
    fig_donut = go.Figure(data=[go.Pie(
        labels=teams,
        values=predicted_probabilities,
        hole=0.4,
        marker=dict(colors=["#00A86B", "#D72638"]),
        textinfo='label+percent'
    )])
    fig_donut.update_layout(title_text="📊 Match Win Probability")
    st.plotly_chart(fig_donut)


def start_ngrok():
    url = ngrok.connect(8501).public_url
    print(f"📢 Open this URL to access your Streamlit app: {url}")

if __name__ == "__main__":
    start_ngrok()
