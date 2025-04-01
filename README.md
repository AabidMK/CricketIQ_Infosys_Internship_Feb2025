# CricketIQ_Infosys_Internship_Feb2025
The Indian Premier League (IPL) is a popular Twenty20 cricket tournament held annually in India. This project aims to perform exploratory data analysis and create data visualizations on an IPL dataset.

This project predicts the winner of an IPL match using historical data and machine learning models. It takes match details as input and provides winning probabilities using a trained Random Forest model.

📌 GitHub Repository: CricketIQ_Infosys_Internship_Feb2025

📂 Project Structure

graphql
Copy
Edit
📂 CricketIQ_Infosys_Internship_Feb2025/

│── 📜 README.md                # Project documentation  
│── 📜 requirements.txt         # List of dependencies  
│── 📂 datasets/                 # Raw datasets  
│   ├── matches.csv              # IPL match data  
│   ├── deliveries.csv           # Ball-by-ball data  
│── 📂 models/                   # Trained model & encoders  
│   ├── final_rf_model.pkl       # Trained model  
│   ├── le_team.pkl              # Label encoder for teams  
│   ├── le_venue.pkl             # Label encoder for venues  
│── 📂 notebooks/                # Jupyter Notebooks for EDA & training  
│   ├── eda_data_processing.ipynb  # Data preprocessing & EDA  
│   ├── model_training.ipynb     # Model training  
│── 📂 scripts/                  # Python scripts for model & UI  
│   ├── streamlit_app.py         # Streamlit UI script  
│   ├── train_model.py           # Script for model training  

⚙️ Setup & Installation

1️⃣ Clone the Repository

bash
Copy
Edit
git clone https://github.com/AabidMK/CricketIQ_Infosys_Internship_Feb2025.git  
cd CricketIQ_Infosys_Internship_Feb2025  

2️⃣ Create & Activate Virtual Environment

bash
Copy
Edit
python -m venv venv  
source venv/bin/activate  # Mac/Linux  
venv\Scripts\activate     # Windows  

3️⃣ Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt  
📜 requirements.txt
txt
Copy
Edit
numpy==2.1.3
pandas==2.1.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.2
streamlit==1.33.0
joblib==1.3.2
pickle-mixin==1.0.2
matplotlib.pyplot==3.8.2

📌 Dependencies needed for:

Data processing (numpy, pandas)

Model training (scikit-learn, joblib, pickle-mixin)

Visualization (matplotlib.pyplot, seaborn)

Web UI (streamlit)

📊 Data Processing & EDA
Dataset Used:

matches.csv: Contains match-level details.

deliveries.csv: Contains ball-by-ball details.

Processing Steps:

Cleaned missing values.

Encoded categorical features (teams, venues).

Calculated additional features like current run rate (CRR) & required run rate (RRR).

Used matplotlib.pyplot and seaborn for data visualization.

📌 Check: eda_data_processing.ipynb

🛠 Model Training
Model: Random Forest Classifier

Features Used:

Batting Team, Bowling Team, Venue, Total Runs, Wickets, Overs, Target (if 2nd innings), Run Rate Impact.

Data Encoding: Used Label Encoding (le_team.pkl, le_venue.pkl).

Data Visualization: Used matplotlib.pyplot for feature importance plots.

Model trained & saved as final_rf_model.pkl.

📌 Check: model_training.ipynb

🎯 Streamlit UI
Run the Streamlit app to predict the match outcome in real time.

bash
Copy
Edit
streamlit run scripts/streamlit_app.py  
🔹 UI Features
Select Batting Team, Bowling Team, Venue.

Input Runs, Wickets, Overs (and Target for 2nd innings).

Computes CRR & RRR automatically.

Displays winning probability using a bar chart (Matplotlib + Streamlit).

📌 Check: streamlit_app.py

📊 Prediction Results
Input Feature	Example Value
Batting Team	CSK
Bowling Team	MI
Venue	Wankhede Stadium
Total Runs	160
Wickets Lost	4
Overs Completed	15
Target (if 2nd Innings)	180
Prediction	CSK 65% - MI 35%
Bar Chart Example:

python
Copy
Edit
import matplotlib.pyplot as plt

teams = ["CSK", "MI"]
probabilities = [65, 35]

plt.bar(teams, probabilities, color=["yellow", "blue"])
plt.xlabel("Teams")
plt.ylabel("Win Probability (%)")
plt.title("IPL Match Prediction Result")
plt.show()
📌 This is implemented in streamlit_app.py, where winning probabilities are displayed using Matplotlib in Streamlit.

📝 Future Enhancements
Use Deep Learning (LSTMs) for better predictions.

Add Live API Integration for real-time match updates.

Improve UI with dynamic graphs.

👥 Team & Contributors
This project is a team collaboration, guided by our mentor and developed by a dedicated group of contributors.

👨‍🏫 Mentor
Aabid MK (INFOSYS SPRINGBOARD)

👨‍💻 Team Members
Abhishek Jyoti
Akash V
Amol Deshmukh
Anshika Sharma
Anusuyya GJ
Arbitha Reddy Gaddam
Ashish Patel
Atmakuri Lahari
Ayshwarya Karthikeyan
Balaji Khavane
Bhavya Sri
Dhayanish S
Dintakurthi Aakansha Sai
Durga Bhavani
Durga Jaya Malleswari Tommandru
Vegavathi GB
Giriraj SA
Harsha Vardini Gopireddy
Jaibalaji
Jyothirmayi Ramisetti
Keerthiga Devi M
Nagesh Guguloth
Prince
Tamil Selvi P
Vaishnavi Ravikumar
Vignesh A

📌 Feel free to contribute! 🚀


