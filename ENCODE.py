import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\Giriraj S A\Downloads\merged_data.csv")

# Print columns to verify
print("✅ Available columns:", df.columns.tolist())

# Create LabelEncoders
le_team = LabelEncoder()
le_venue = LabelEncoder()

# Fit encoders on unique team names and venue names
df["team1_encoded"] = le_team.fit_transform(df["batting_team"])
df["team2_encoded"] = le_team.transform(df["bowling_team"])
df["venue_encoded"] = le_venue.fit_transform(df["venue"])

# Save the encoders
joblib.dump(le_team, r"C:\Users\Giriraj S A\Downloads\le_team.pkl")
joblib.dump(le_venue, r"C:\Users\Giriraj S A\Downloads\le_venue.pkl")

print("✅ Encoders saved successfully! 🎯")
