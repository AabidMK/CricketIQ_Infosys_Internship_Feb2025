import streamlit as st
import joblib
import numpy as np

# Load label encoders and model
le_team = joblib.load(r'C:\Users\DELL\OneDrive\Desktop\match\le_team.pkl')
le_venue = joblib.load(r'C:\Users\DELL\OneDrive\Desktop\match\le_venue.pkl')
model = joblib.load(r'C:\Users\DELL\OneDrive\Desktop\match\final_rf_model.pkl')

teams = le_team.classes_.tolist()
venues = le_venue.classes_.tolist()

st.set_page_config(page_title="Win Probability Predictor", layout="wide")

st.markdown(
    """
    <style>
        body {
            background-color: #001F3F;
            color: white;
        }
        .main {
            background-color: #001F3F;
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .stApp {
            background-color: #001F3F;
        }
        .title-text {
            font-size: 32px;
            font-weight: bold;
            color: #FFD700; /* Gold */
            text-align: center;
        }
        .section-title {
            font-size: 26px;
            font-weight: bold;
            color: #FF4136; /* Red */
            text-align: center;
        }
        .metric-box {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: white;
            background-color: rgba(255, 255, 255, 0.2);
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
        }
        .highlight {
            font-size: 24px;
            font-weight: bold;
            color: #39FF14; /* Neon Green */
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title-text'>🏏 Live Win Probability Prediction</div>", unsafe_allow_html=True)
st.sidebar.header("🏟️ Match Input Details")

inning = st.sidebar.selectbox("Innings", options=[1, 2])
batting_team = st.sidebar.selectbox("🏏 Batting Team", teams)
bowling_team = st.sidebar.selectbox("🎯 Bowling Team", [team for team in teams if team != batting_team])
venue = st.sidebar.selectbox("📍 Venue", venues)

cum_runs = st.sidebar.number_input("Total Runs", min_value=0, value=54, step=1)
cum_wickets = st.sidebar.number_input("Wickets Lost", min_value=0, value=3, step=1)
overs_completed = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)

target = st.sidebar.number_input("🎯 Target Score", min_value=0, value=160, step=1) if inning == 2 else 0

current_rr = cum_runs / overs_completed if overs_completed > 0 else 0
required_rr = (target - cum_runs) / (20 - overs_completed) if inning == 2 and (20 - overs_completed) > 0 else 0

st.markdown("<div class='section-title'>📊 Match Overview</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box'>🏏 <b>Batting Team</b><br>{batting_team}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'>🎯 <b>Bowling Team</b><br>{bowling_team}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'>📍 <b>Venue</b><br>{venue}</div>", unsafe_allow_html=True)

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-box'>🏏 <b>Total Runs</b><br>{cum_runs}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'>❌ <b>Wickets Lost</b><br>{cum_wickets}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'>⏳ <b>Overs Completed</b><br>{overs_completed}</div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-box'>🎯 <b>Target</b><br>{target if inning == 2 else 'N/A'}</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>📈 Run Rates</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div class='metric-box'>🔥 <b>Current Run Rate</b><br>{round(current_rr, 2)}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'>🚀 <b>Required Run Rate</b><br>{round(required_rr, 2) if inning == 2 else 'N/A'}</div>", unsafe_allow_html=True)

input_data = np.array([[inning, le_team.transform([batting_team])[0],
                        le_team.transform([bowling_team])[0], le_venue.transform([venue])[0],
                        cum_runs, cum_wickets, overs_completed, target if inning == 2 else 0,
                        current_rr]]).reshape(1, -1)

win_prob = model.predict_proba(input_data)[0]  # Assuming model supports predict_proba()
batting_team_win_prob = round(win_prob[1] * 100, 2)
bowling_team_win_prob = round(win_prob[0] * 100, 2)
predicted_winner = batting_team if batting_team_win_prob > bowling_team_win_prob else bowling_team

st.markdown("<div class='section-title'>🔮 Win Probability Prediction</div>", unsafe_allow_html=True)

st.markdown(f"<div class='metric-box' style='color: #FFD700;'>🏏 <b>{batting_team} Winning Probability:</b> {batting_team_win_prob}%</div>", unsafe_allow_html=True)
st.progress(batting_team_win_prob / 100)

st.markdown(f"<div class='metric-box' style='color: #FF4136;'>🎯 <b>{bowling_team} Winning Probability:</b> {bowling_team_win_prob}%</div>", unsafe_allow_html=True)
st.progress(bowling_team_win_prob / 100)
import plotly.graph_objects as go

import streamlit as st
import plotly.graph_objects as go

# Define new colors
batting_team_color = "#de4c3c"  # Violet
bowling_team_color = "#6ee344"  # Dark Violet
background_color = "#000000"  # Black

# Win Probability Section
st.markdown("<div class='section-title'>🔮 Win Probability Prediction</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])  # Arrange metric boxes & chart neatly

with col1:
    st.markdown(f"""
        <div class='metric-box' style='color: {batting_team_color}; background-color: rgba(255, 255, 255, 0.2);'>
        🏏 <b>{batting_team} Winning Probability:</b> {batting_team_win_prob}%</div>
        """, unsafe_allow_html=True)
    st.progress(batting_team_win_prob / 100)

    st.markdown(f"""
        <div class='metric-box' style='color: {bowling_team_color}; background-color: rgba(255, 255, 255, 0.2);'>
        🎯 <b>{bowling_team} Winning Probability:</b> {bowling_team_win_prob}%</div>
        """, unsafe_allow_html=True)
    st.progress(bowling_team_win_prob / 100)

with col2:
    # Create a stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Win Probability'],
        y=[batting_team_win_prob],
        name=f"{batting_team}",
        marker_color=batting_team_color
    ))
    fig.add_trace(go.Bar(
        x=['Win Probability'],
        y=[bowling_team_win_prob],
        name=f"{bowling_team}",
        marker_color=bowling_team_color
    ))

    # Layout settings
    fig.update_layout(
        barmode='stack',
        title="Win Probability Distribution",
        xaxis_title="",
        yaxis_title="Probability (%)",
        showlegend=True,
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor=background_color,  # Set black background
        paper_bgcolor=background_color,  # Set black background
        font=dict(color="white")  # Set text color to white for visibility
    )

    # Display stacked bar chart
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"<h3 style='color: #28A745; text-align: center;'>🏆 Predicted Winner: {predicted_winner}</h3>", unsafe_allow_html=True)

