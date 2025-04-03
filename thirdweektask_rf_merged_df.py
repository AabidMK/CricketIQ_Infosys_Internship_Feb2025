#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


match_df=pd.read_csv(r"C:\Users\Giriraj S A\Downloads\matches.csv")


# In[4]:


match_df.info()


# In[5]:


match_df.isnull().sum()


# In[6]:


print(match_df['city'])


# In[7]:


match_df['city'] = match_df['city'].fillna("no update")


# In[8]:


print(match_df['method'])


# In[9]:


match_df = match_df.drop(columns=['method'])


# In[10]:


match_df['winner'].fillna("no result")


# In[11]:


match_df['player_of_match'].fillna("no player")


# In[12]:


delivery_df=pd.read_csv(r"c:\Users\Giriraj S A\Downloads\deliveries (2).csv")


# In[13]:


delivery_df.info()


# In[14]:


delivery_df.isnull().sum()


# In[15]:


delivery_df['extras_type'].fillna("no extras")


# In[16]:


delivery_df['dismissal_kind'].fillna("no wicket")


# In[17]:


if 'over' not in delivery_df.columns or 'ball' not in delivery_df.columns:
    raise ValueError("Missing 'over' or 'ball' column in delivery_df")



# In[18]:


delivery_df['Balls_Bowled'] = (delivery_df['over'] * 6 + (delivery_df['ball'] - 1)).astype(int)



# In[19]:


delivery_df['overs_completed'] = delivery_df['Balls_Bowled'] // 6 + (delivery_df['Balls_Bowled'] % 6) / 10



# In[20]:


delivery_df['overs_completed'] = delivery_df['overs_completed'].apply(lambda x: f"{int(x)}.{int((x - int(x)) * 10)}")


# In[21]:


print(delivery_df[['over', 'ball', 'Balls_Bowled', 'overs_completed']].head())  # Debug output


# In[22]:


import pandas as pd
if 'over' not in delivery_df.columns or 'ball' not in delivery_df.columns:
    raise ValueError("Missing 'over' or 'ball' column in delivery_df")
delivery_df['Balls_Bowled'] = (delivery_df['over'] * 6 + (delivery_df['ball'] - 1)).astype(int)
delivery_df['cumulative_runs'] = delivery_df.groupby(['match_id', 'inning', 'batting_team'])['total_runs'].cumsum()
delivery_df['CRR'] = (delivery_df['cumulative_runs'] / (delivery_df['Balls_Bowled'] / 6)).round(2)
delivery_df['RRR'] = None  
target_scores = delivery_df[delivery_df['inning'] == 1].groupby('match_id')['cumulative_runs'].max()
second_innings_mask = delivery_df['inning'] == 2

delivery_df.loc[second_innings_mask, 'RRR'] = (
    (delivery_df['match_id'].map(target_scores) - delivery_df['cumulative_runs']) / 
    ((120 - delivery_df['Balls_Bowled']) / 6)
).round(2)
delivery_df['CRR'] = delivery_df['CRR'].apply(lambda x: f"{x:.2f} RPO" if pd.notnull(x) else None)
delivery_df['RRR'] = delivery_df['RRR'].apply(lambda x: f"{x:.2f} RPO" if pd.notnull(x) else None)


# In[23]:


print(delivery_df['CRR'])


# In[24]:


print(match_df.columns)

match_df.rename(columns={"id": "match_id"}, inplace=True)


# In[25]:


merged_df = pd.merge(delivery_df, match_df, on="match_id", how="inner")



# In[26]:


print(match_df.columns)

match_df.rename(columns={"id": "match_id"}, inplace=True)



# In[27]:


print(match_df.columns)



# In[28]:


merged_df = pd.merge(delivery_df, match_df, on="match_id", how="inner")


# In[29]:


print(merged_df.columns)


# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
if 'merged_df' not in globals():
    raise ValueError("Merged dataset (merged_df) is not defined. Ensure it's created properly.")

encoder = LabelEncoder()
merged_df['batting_team_encoded'] = encoder.fit_transform(merged_df['batting_team'])
merged_df['bowling_team_encoded'] = encoder.fit_transform(merged_df['bowling_team'])
if 'venue' in merged_df.columns:
    merged_df['venue_encoded'] = encoder.fit_transform(merged_df['venue'])
elif 'city' in merged_df.columns:
    merged_df['city_encoded'] = encoder.fit_transform(merged_df['city'])
merged_df['win'] = (merged_df['batting_team'] == merged_df['winner']).astype(int)
if 'cumulative_wickets' not in merged_df.columns:
    merged_df['cumulative_wickets'] = merged_df.groupby(['match_id', 'inning', 'batting_team'])['is_wicket'].cumsum()
rename_map = {
    'cumulative_runs': 'cum_runs',
    'CRR': 'current_run_rate',
    'RRR': 'required_run_rate'
}

for old_col, new_col in rename_map.items():
    if old_col in merged_df.columns and new_col not in merged_df.columns:
        merged_df.rename(columns={old_col: new_col}, inplace=True)
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

final_features = [
    'inning', 'cum_runs', 'cumulative_wickets', 'current_run_rate', 'required_run_rate', 'target_runs',
    'batting_team_encoded', 'bowling_team_encoded', 'win'
]

# Add venue or city encoding
if 'venue_encoded' in merged_df.columns:
    final_features.append('venue_encoded')
elif 'city_encoded' in merged_df.columns:
    final_features.append('city_encoded')

# Create final dataset
final_df = merged_df[final_features].drop_duplicates()

# Train-test split
X = final_df.drop(columns=['win'])
y = final_df['win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display first few rows of the final dataset
print(final_df.head())


# In[31]:


print(final_df.columns)


# In[32]:


team_mapping = {
    'Mumbai Indians': 0,
    'Chennai Super Kings': 1,
    'Delhi Daredevils': 2,  # Same ID even after renaming
    'Delhi Capitals': 2,     # Ensuring renamed teams get the same number
    'Gujarat Lions': 3,
    'Rajasthan Royals': 4,
    'Sunrisers Hyderabad': 5,
    'Royal Challengers Bangalore': 6,
    'Punjab Kings': 7,
    'Kolkata Knight Riders': 8,
    'Lucknow Super Giants': 9
}

# Encode batting and bowling teams using the mapping
merged_df['batting_team_encoded'] = merged_df['batting_team'].map(team_mapping)
merged_df['bowling_team_encoded'] = merged_df['bowling_team'].map(team_mapping)


# In[ ]:





# In[33]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
if 'merged_df' not in globals():
    raise ValueError("Merged dataset (merged_df) is not defined. Ensure it's created properly.")
if 'current_run_rate' in merged_df.columns:
    merged_df['current_run_rate'] = (
        merged_df['current_run_rate']
        .astype(str)
        .str.replace(" RPO", "", regex=False)
        .replace("None", np.nan) 
        .astype(float) 
    )

if 'required_run_rate' in merged_df.columns:
    merged_df['required_run_rate'] = (
        merged_df['required_run_rate']
        .astype(str)
        .str.replace(" RPO", "", regex=False)
        .replace("None", np.nan)
        .astype(float) 
    )
numerical_features = ['current_run_rate', 'required_run_rate', 'cumulative_runs', 'cumulative_wickets', 'target_runs']
for col in numerical_features:
    if col in merged_df.columns:
        merged_df[col].fillna(merged_df[col].median(), inplace=True)
encoder = LabelEncoder()
if 'batting_team' in merged_df.columns:
    merged_df['batting_team_encoded'] = encoder.fit_transform(merged_df['batting_team'])

if 'bowling_team' in merged_df.columns:
    merged_df['bowling_team_encoded'] = encoder.fit_transform(merged_df['bowling_team'])

if 'venue' in merged_df.columns:
    merged_df['venue_encoded'] = encoder.fit_transform(merged_df['venue'])
elif 'city' in merged_df.columns:
    merged_df['city_encoded'] = encoder.fit_transform(merged_df['city'])
if 'batting_team' in merged_df.columns and 'winner' in merged_df.columns:
    merged_df['win'] = (merged_df['batting_team'] == merged_df['winner']).astype(int)
else:
    raise KeyError("Columns 'batting_team' and/or 'winner' are missing. Cannot determine win label.")
required_features = [
    'inning', 'cumulative_runs', 'cumulative_wickets', 'current_run_rate', 'required_run_rate', 'target_runs',
    'batting_team_encoded', 'bowling_team_encoded', 'win'
]
final_features = [col for col in required_features if col in merged_df.columns]
if 'venue_encoded' in merged_df.columns:
    final_features.append('venue_encoded')
elif 'city_encoded' in merged_df.columns:
    final_features.append('city_encoded')
if len(final_features) < 2:
    raise ValueError("Not enough features available for training. Available columns: ", merged_df.columns)
final_df = merged_df[final_features].drop_duplicates()
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.fillna(final_df.median(), inplace=True) 
final_df.fillna(final_df.mode().iloc[0], inplace=True)
X = final_df.drop(columns=['win'])
y = final_df['win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Evaluate model performance
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


# In[34]:


import pickle

model_path = 'trained_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf, file)

print(f"Model saved to {model_path}")


# In[35]:


import pickle

model_path = 'trained_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf, file)

print(f"Model saved to {model_path}")


# In[36]:


import os
print(os.path.exists('trained_model.pkl'))


# In[37]:


import pickle

model_path = 'trained_model.pkl'
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)


# In[38]:


import os
print(os.getcwd())


# In[39]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
if 'merged_df' not in globals():
    raise ValueError("Merged dataset (merged_df) is not defined. Ensure it's created properly.")
if 'current_run_rate' in merged_df.columns:
    merged_df['current_run_rate'] = (
        merged_df['current_run_rate']
        .astype(str)
        .str.replace(" RPO", "", regex=False)
        .replace("None", np.nan) 
        .astype(float)  
    )

if 'required_run_rate' in merged_df.columns:
    merged_df['required_run_rate'] = (
        merged_df['required_run_rate']
        .astype(str)
        .str.replace(" RPO", "", regex=False)
        .replace("None", np.nan)
        .astype(float)
    )
numerical_features = ['current_run_rate', 'required_run_rate', 'cumulative_runs', 'cumulative_wickets', 'target_runs']
for col in numerical_features:
    if col in merged_df.columns:
        merged_df[col].fillna(merged_df[col].median(), inplace=True)
encoder = LabelEncoder()
if 'batting_team' in merged_df.columns:
    merged_df['batting_team_encoded'] = encoder.fit_transform(merged_df['batting_team'])

if 'bowling_team' in merged_df.columns:
    merged_df['bowling_team_encoded'] = encoder.fit_transform(merged_df['bowling_team'])

if 'venue' in merged_df.columns:
    merged_df['venue_encoded'] = encoder.fit_transform(merged_df['venue'])
elif 'city' in merged_df.columns:
    merged_df['city_encoded'] = encoder.fit_transform(merged_df['city'])
if 'batting_team' in merged_df.columns and 'winner' in merged_df.columns:
    merged_df['win'] = (merged_df['batting_team'] == merged_df['winner']).astype(int)
else:
    raise KeyError("Columns 'batting_team' and/or 'winner' are missing. Cannot determine win label.")
required_features = [
    'inning', 'cumulative_runs', 'cumulative_wickets', 'current_run_rate', 'required_run_rate', 'target_runs',
    'batting_team_encoded', 'bowling_team_encoded', 'win'
]
final_features = [col for col in required_features if col in merged_df.columns]
if 'venue_encoded' in merged_df.columns:
    final_features.append('venue_encoded')
elif 'city_encoded' in merged_df.columns:
    final_features.append('city_encoded')
if len(final_features) < 2:
    raise ValueError("Not enough features available for training. Available columns: ", merged_df.columns)
final_df = merged_df[final_features].drop_duplicates()
final_df.replace([np.inf, -np.inf], np.nan, inplace=True) 
final_df.fillna(final_df.median(), inplace=True) 
final_df.fillna(final_df.mode().iloc[0], inplace=True) 
X = final_df.drop(columns=['win'])
y = final_df['win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


# In[40]:


import streamlit as st


# In[41]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load Data
@st.cache_data
def load_data():
    match_df = pd.read_csv(r"C:\\Users\\Giriraj S A\\Downloads\\matches.csv")
    delivery_df = pd.read_csv(r"C:\\Users\\Giriraj S A\\Downloads\\deliveries.csv")

    # Fill missing values
    match_df['city'] = match_df['city'].fillna("no update")
    match_df['winner'] = match_df['winner'].fillna("no result")
    match_df['player_of_match'] = match_df['player_of_match'].fillna("no player")
    delivery_df['extras_type'] = delivery_df['extras_type'].fillna("no extras")
    delivery_df['dismissal_kind'] = delivery_df['dismissal_kind'].fillna("no wicket")

    # Drop irrelevant columns
    match_df = match_df.drop(columns=['method'])

    # Merge data
    merged_df = pd.merge(delivery_df, match_df, on='match_id', how='inner')

    # Compute additional features
    merged_df['Balls_Bowled'] = (merged_df['over'] * 6 + (merged_df['ball'] - 1)).astype(int)
    merged_df['overs_completed'] = merged_df['Balls_Bowled'] // 6 + (merged_df['Balls_Bowled'] % 6) / 6
    merged_df['current_run_rate'] = merged_df['total_runs'] / merged_df['overs_completed']
    merged_df['required_run_rate'] = (merged_df['target'] - merged_df['total_runs']) / (20 - merged_df['overs_completed'])

    # Create win/loss label
    merged_df['win'] = (merged_df['batting_team'] == merged_df['winner']).astype(int)

    return merged_df

# Train Model
def train_model(df):
    encoder = LabelEncoder()
    df['batting_team_encoded'] = encoder.fit_transform(df['batting_team'])
    df['bowling_team_encoded'] = encoder.fit_transform(df['bowling_team'])

    X = df[['inning', 'total_runs', 'Balls_Bowled', 'current_run_rate', 'required_run_rate', 'overs_completed',
             'batting_team_encoded', 'bowling_team_encoded']]
    y = df['win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# Load data and train model
st.title("🏏 Cricket Match Outcome Predictor")
data = load_data()

if st.sidebar.button("Train Model"):
    model = train_model(data)
    st.sidebar.success("Model trained successfully! ✅")

# User Inputs
batting_team = st.selectbox("Select Batting Team", data['batting_team'].unique())
bowling_team = st.selectbox("Select Bowling Team", data['bowling_team'].unique())
inning = st.number_input("Inning", min_value=1, max_value=2)
total_runs = st.number_input("Total Runs Scored", min_value=0)
balls_bowled = st.number_input("Balls Bowled", min_value=0)
current_run_rate = st.number_input("Current Run Rate", min_value=0.0)
required_run_rate = st.number_input("Required Run Rate", min_value=0.0)
overs_completed = balls_bowled // 6 + (balls_bowled % 6) / 6

if st.button("Predict Outcome"):
    input_data = pd.DataFrame({
        'inning': [inning],
        'total_runs': [total_runs],
        'Balls_Bowled': [balls_bowled],
        'current_run_rate': [current_run_rate],
        'required_run_rate': [required_run_rate],
        'overs_completed': [overs_completed],
        'batting_team_encoded': [data.loc[data['batting_team'] == batting_team, 'batting_team_encoded'].values[0]],
        'bowling_team_encoded': [data.loc[data['bowling_team'] == bowling_team, 'bowling_team_encoded'].values[0]],
    })

    prediction = model.predict(input_data)[0]
    result = "Batting team will win! 🏆" if prediction == 1 else "Batting team will lose! ❌"
    st.write(f"### 🎯 Prediction: {result}")

# Display DataFrame (Optional)
if st.checkbox("Show Raw Data"):
    st.write(data.head(20))


# In[ ]:




