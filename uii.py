import streamlit as st
import pandas as pd
import pickle  

# Load the trained model
@st.cache_data
def load_model():
    try:
        with open("pkl file/randomforest_classifier.pkl", "rb") as f:
            model = pickle.load(f)

        if not hasattr(model, "predict"):
            st.error("❌ Invalid ML model. Please check the model file.")
            return None
        return model

    except Exception as e:
        st.error(f"🚨 Error loading model: {str(e)}")
        return None

model = load_model()

# Sidebar Inputs
st.sidebar.header("Match Input Details")

innings = st.sidebar.selectbox("Innings", [1, 2])
cumulative_runs = st.sidebar.number_input("Cumulative Runs", min_value=0, value=50)
cumulative_wickets = st.sidebar.number_input("Cumulative Wickets", min_value=0, max_value=10, value=2)
overs_completed = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=50.0, step=0.1, value=7.0)
target = st.sidebar.number_input("Target Score", min_value=0, value=150)  # ✅ Include 'target' as expected

batting_team = st.sidebar.selectbox("Batting Team", ["Chennai Super Kings", "Deccan Chargers"])
bowling_team = st.sidebar.selectbox("Bowling Team", ["Deccan Chargers", "Chennai Super Kings"])
venue = st.sidebar.text_input("Venue", "Arun Jaitley Stadium, Delhi")

# Calculate Derived Features
current_run_rate = cumulative_runs / overs_completed if overs_completed > 0 else 0.0
required_run_rate = (target - cumulative_runs) / (50 - overs_completed) if overs_completed < 50 else 0.0  # ✅ Compute 'required_run_rate'

# Prepare Input Data with Correct Feature Names
input_data = pd.DataFrame({
    "match_id": [0],  
    "inning": [innings],  
    "cum_runs": [cumulative_runs],  
    "cum_wickets": [cumulative_wickets],  
    "current_run_rate": [current_run_rate],  
    "required_run_rate": [required_run_rate],  
    "target": [target],  
    "batting_team": [batting_team],
    "bowling_team": [bowling_team],
    "venue_canonical": [venue]  
})

# Convert categorical values to numerical
team_mapping = {
    "Chennai Super Kings": 0,
    "Deccan Chargers": 1
}

venue_mapping = {
    "Arun Jaitley Stadium, Delhi": 0  # Add more venues if required
}

input_data["batting_team"] = input_data["batting_team"].map(team_mapping)
input_data["bowling_team"] = input_data["bowling_team"].map(team_mapping)
input_data["venue_canonical"] = input_data["venue_canonical"].map(venue_mapping).fillna(-1)  # ✅ Match expected column name

# Make Prediction
if model is not None:
    try:
        prediction = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else model.predict(input_data)[0]
        win_prob_batting = prediction[1] * 100 if hasattr(model, "predict_proba") else prediction * 100
        win_prob_bowling = 100 - win_prob_batting

        winner = batting_team if win_prob_batting > win_prob_bowling else bowling_team

        st.subheader("📊 Prediction Results")
        st.write(f"🏆 **{batting_team} Winning Probability:** {win_prob_batting:.2f}%")
        st.write(f"🎯 **{bowling_team} Winning Probability:** {win_prob_bowling:.2f}%")
        st.success(f"✅ **Predicted Winner:** {winner}")

    except Exception as e:
        st.error(f"⚠️ Error in Prediction: {str(e)}")
