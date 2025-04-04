import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

batting_encoder_path = r"C:\Users\Ayshu\label_batting.pkl"
bowling_encoder_path = r"C:\Users\Ayshu\label_bowling.pkl"
venue_encoder_path = r"C:\Users\Ayshu\label_venue.pkl"
model_path = r"C:\Users\Ayshu\final_rf_model.pkl"
dataset_path = r"C:\Users\Ayshu\merged_dataset.csv"


def load_encoders():
    encoders = {}
    try:
        if os.path.exists(batting_encoder_path):
            encoders["batting"] = joblib.load(batting_encoder_path)
        else:
            st.error("❌ Batting team encoder file is missing!")

        if os.path.exists(bowling_encoder_path):
            encoders["bowling"] = joblib.load(bowling_encoder_path)
        else:
            st.error("❌ Bowling team encoder file is missing!")

        if os.path.exists(venue_encoder_path):
            encoders["venue"] = joblib.load(venue_encoder_path)
        else:
            st.error("❌ Venue encoder file is missing!")

        

    except Exception as e:
        st.error(f"❌ Error loading encoders: {e}")
        st.stop()
    
    return encoders

encoders = load_encoders()

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(model_path):
            st.error("❌ Model file not found! Please check the file path.")
            st.stop()
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

try:
    if not os.path.exists(dataset_path):
        st.error("❌ Dataset file not found! Please check the file path.")
        st.stop()

    df = pd.read_csv(dataset_path)
    batting_teams = sorted(df["batting_team"].unique().tolist())
    bowling_teams = sorted(df["bowling_team"].unique().tolist())
    venues = sorted(df["venue"].unique().tolist())
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

st.title("🏏 Live Cricket Match Prediction")
st.sidebar.header("🏆 Match Input Details")

with st.sidebar.form("match_form"):
    innings = st.selectbox("Select Innings", [1, 2])
    batting_team = st.selectbox("Select Batting Team", batting_teams)
    bowling_team = st.selectbox("Select Bowling Team", bowling_teams)
    venue = st.selectbox("Select Venue", venues)

    overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, step=0.1)
    runs = st.number_input("Cumulative Runs", min_value=0, step=1)
    wickets = st.number_input("Cumulative Wickets", min_value=0, max_value=10, step=1)

    target = None
    if innings == 2:
        target = st.number_input("Target Score", min_value=runs + 1, step=1)
    else:
        target = 0  # Set default for first innings

    submit_button = st.form_submit_button("Predict Outcome")

if submit_button:
    try:
        batting_team_encoded = encoders["batting"].transform([batting_team])[0]
        bowling_team_encoded = encoders["bowling"].transform([bowling_team])[0]
        venue_encoded = encoders["venue"].transform([venue])[0]

        current_run_rate = runs / overs_completed if overs_completed > 0 else 0
        required_run_rate = ((target - runs) / (50 - overs_completed)) if (innings == 2 and target > 0) else 0
        
        input_features = np.array([[innings, runs, wickets, current_run_rate, required_run_rate,
                                    target, batting_team_encoded, bowling_team_encoded,
                                    venue_encoded, overs_completed]])

        with st.spinner("🔄 Predicting outcome... Please wait!"):
            prediction = model.predict_proba(input_features.reshape(1, -1))[0]

        # ✅ Display Predictions
        st.success("✅ Prediction Successful!")
        st.write(f"📊 **Current Run Rate (CRR):** {current_run_rate} runs/over")
        if innings == 2:
            st.write(f"🔢 **Required Run Rate (RRR):** {required_run_rate} runs/over")
        st.write(f"🏏 **Winning Probability of {batting_team}:** {round(prediction[1] * 100, 2)}%")
        st.write(f"🎯 **Winning Probability of {bowling_team}:** {round(prediction[0] * 100, 2)}%")

    except ValueError:
        st.error("❌ Selected team not found in the encoder file. Please check dataset & encoder!")
        st.write("**Available Teams in Encoder:**", list(encoders["batting"].classes_))
        st.stop()
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
