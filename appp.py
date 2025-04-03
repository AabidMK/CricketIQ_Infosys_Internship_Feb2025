import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Path to the .pkl file on your local system
model_path = r"C:\Users\bhavy\Downloads\final_rf_model (1).pkl"

# Check if the model file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Error: Model file not found at {model_path}")
    st.stop()

# Team Encoding (must match training data)
team_encoding = {
    'Mumbai Indians': 0,
    'Chennai Super Kings': 1,
    'Royal Challengers Bengaluru': 2,
    'Kolkata Knight Riders': 3,
    'Delhi Capitals': 4,
    'Sunrisers Hyderabad': 5,
    'Punjab Kings': 6,
    'Rajasthan Royals': 7,
    'Gujarat Titans': 8,
    'Lucknow Super Giants': 9
}

# Stadium Encoding
stadium_encoding = {
    "Wankhede Stadium": 0,
    "Eden Gardens": 1,
    "M. Chinnaswamy Stadium": 2,
    "Arun Jaitley Stadium": 3,
    "MA Chidambaram Stadium": 4,
    "Narendra Modi Stadium": 5,
    "Punjab Cricket Association Stadium": 6,
    "Rajiv Gandhi International Stadium": 7,
    "Sawai Mansingh Stadium": 8,
    "BRSABV Ekana Cricket Stadium": 9
}

# Function to preprocess match data
def preprocess_match_data(batting_team, bowling_team, inning, runs, wickets, crr, rrr, target, venue_encoded):
    match_df = pd.DataFrame([{
        'inning': inning,
        'cumulative_runs': runs,
        'cumulative_wickets': wickets,
        'current_run_rate': crr,
        'required_run_rate': rrr,
        'target_runs': target,
        'batting_team_encoded': team_encoding[batting_team],
        'bowling_team_encoded': team_encoding[bowling_team],
        'venue_encoded': venue_encoded
    }])
    return match_df

# Streamlit App
def app():
    st.title("🏏 IPL Match Winner Prediction")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Match Input")
        innings = st.radio("Select Innings", (1, 2))
        batting_team = st.selectbox("Select Batting Team", list(team_encoding.keys()))
        bowling_team = st.selectbox("Select Bowling Team", list(team_encoding.keys()))
        venue_name = st.selectbox("Select Venue", list(stadium_encoding.keys()))
        venue_encoded = stadium_encoding[venue_name]
        runs = st.number_input("Enter Total Runs Scored", min_value=0, step=1)
        wickets = st.number_input("Enter Total Wickets Lost", min_value=0, max_value=10, step=1)
        overs_completed = st.number_input("Enter Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
        
        target = 0
        crr = 0
        rrr = 0
        if innings == 2:
            target = st.number_input("Enter Target Score", min_value=1, step=1)
            if overs_completed > 0:
                crr = runs / overs_completed
            if overs_completed < 20 and target > runs:
                rrr = (target - runs) / (20 - overs_completed)
            
            st.write(f"📊 **Current Run Rate (CRR): {crr:.2f}**")
            st.write(f"🔥 **Required Run Rate (RRR): {rrr:.2f}**")
        
        predict_button = st.button("🔮 Predict Winner")
    
    # Prediction and display results in main content
    if predict_button:
        try:
            match_data = preprocess_match_data(batting_team, bowling_team, innings, runs, wickets, crr, rrr, target, venue_encoded)
            prediction = model.predict(match_data)[0]
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"🏆 **Prediction: {batting_team} is likely to WIN!**")
            else:
                st.error(f"❌ **Prediction: {bowling_team} might WIN!**")
            
            # Match statistics
            st.subheader("Match Statistics")
            st.write(f"📊 **Total Runs Scored by {batting_team}: {runs}**")
            st.write(f"🔴 **Total Wickets Lost by {batting_team}: {wickets}**")
            st.write(f"⏱️ **Overs Completed: {overs_completed:.1f}**")
            if innings == 2:
                st.write(f"🎯 **Target Score: {target}**")
                st.write(f"⚡ **Current Run Rate (CRR): {crr:.2f}**")
                st.write(f"🚨 **Required Run Rate (RRR): {rrr:.2f}**")
            
            # Venue Information
            st.subheader("Venue Information")
            st.write(f"🏟️ **Venue: {venue_name}**")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    # Footer
    st.markdown("""
    ---
    ⚡ **Enjoy the game!** ⚡
    **Note**: This prediction model is based on historical match data and may not guarantee accurate results for each game.
    """)

# Run the app
if __name__ == "__main__":
    app()
