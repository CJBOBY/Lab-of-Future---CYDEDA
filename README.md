🎬 CYDEDA – Intelligent Movie Recommender System
CYDEDA is an AI-powered movie recommendation system that combines machine learning, clustering, and generative AI to help users discover personalized movie suggestions based on their preferences. The system utilizes the TMDb 5000 Movies Dataset, performs feature engineering, applies KMeans clustering, and leverages Google Gemini to generate insightful summaries.

A smart desktop app built with Python and Tkinter for cinema lovers who want intelligent and intuitive recommendations.

📌 Table of Contents
🚀 Features

🧠 How It Works

💻 GUI Preview

📦 Installation

🛠️ Tech Stack

🧪 Results

🔮 Future Work

📚 References

👨‍💻 Authors

🚀 Features
✅ Multiple filter options (genres, runtime, rating, year range, cast, director, production company, etc.)

🎬 Similar movie recommendations based on user input titles

🧠 KMeans clustering to group similar movies

🎯 Confidence scores for each recommendation

🤖 Google Gemini integration for AI-generated movie summaries

🖥️ Tkinter GUI with genre checkboxes and smart form input

☁️ Data is loaded from a Supabase cloud storage link

🧠 How It Works
1. 🔎 Feature Extraction
Parses metadata such as genres, keywords, cast, director, and composer

Uses ast.literal_eval() for safe parsing

Encodes genres and other fields using one-hot style vectors

2. 🤖 Clustering
Applies KMeans on numerical and binary features:

Genres

Runtime

Vote average

Release year

Assigns each movie to a cluster and calculates a confidence score

3. 📊 Filtering & Ranking
Users can input preferences

Matching movies are filtered and scored

Top 10 recommendations are shown with confidence %

4. 💬 Gemini AI Summary
Gemini 1.5 Flash generates movie overviews from the final list

Adds a natural-language touch to recommendations

💻 GUI Preview
(Add screenshots here if available)

Page 1: App logo and entry point

Page 2: User input panel (left) + Recommendation output (right)

📦 Installation
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/cydeda.git
cd cydeda
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or install manually:

bash
Copy
Edit
pip install pandas numpy scikit-learn pillow google-generativeai
3. Add your Gemini API key
In cydeda.py, replace this line:

python
Copy
Edit
genai.configure(api_key="your-api-key-here")
Or use an environment variable:

bash
Copy
Edit
export GOOGLE_API_KEY=your-key
4. Run the app
bash
Copy
Edit
python cydeda.py
🛠️ Tech Stack
Component	Technology / Library
Language	Python 3.x
GUI	Tkinter
Clustering	scikit-learn (KMeans)
AI Summaries	Google Generative AI (Gemini 1.5)
Data	TMDb 5000 Movies (via Supabase CSV)
Data Handling	Pandas, NumPy, AST, PIL

🧪 Results
Top 10 movie recommendations align closely with user input

Gemini summaries enhance user engagement

Effective use of clustering without requiring historical user ratings

🔮 Future Work
📱 Convert to mobile app using Flutter or Kivy

🌐 Deploy as a web app using Streamlit or Flask

🗣️ Add voice input and TTS output

🤝 Combine with collaborative filtering for hybrid recommendations

🈂️ Support multilingual recommendations

📚 References
TMDB 5000 Dataset (Kaggle)

scikit-learn KMeans

Tkinter GUI Docs

Google Generative AI (Gemini)

👨‍💻 Authors
Cyriac James Boby

Dev Sebastian Joseph

Dana Shein Rebello
👨‍🔬 Developed under Lab of Future, Rajagiri School of Engineering and Technology
