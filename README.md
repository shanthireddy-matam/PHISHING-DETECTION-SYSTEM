# PHISHING-DETECTION-SYSTEM
About To detect phishing websites based on their URLs using hybrid machine learning techniques. Technologies Used: Dataset: Phishing dataset from Kaggle Algorithms: Hybrid machine learning models (e.g., Decision Trees, Random Forests)

Phishing Detection System Using Hybrid Machine Learning
Overview
This project implements a Phishing Detection System using Hybrid Machine Learning techniques based on URL features. It uses multiple classification models to detect phishing websites efficiently.

Dataset
The dataset used is the Phishing Websites Dataset from the UC Irvine Machine Learning Repository, which contains 30 extracted features from legitimate and phishing websites.

Dataset Link
UCI Phishing Websites Dataset
Technologies Used
Python (Machine Learning & Backend)
Flask (Web API)
HTML, CSS, JavaScript (Frontend)
Scikit-learn, NumPy, Pandas (Machine Learning Libraries)
Matplotlib & Seaborn (Data Visualization)
Installation & Setup
Prerequisites
Ensure you have the following installed:

Python 3.8+
Flask
Scikit-learn
NumPy, Pandas, Matplotlib
Jupyter Notebook (for model training)
Installation Steps
Clone the repository:
git clone https://github.com/yourusername/phishing-detection.git
cd phishing-detection
Install dependencies:
pip install -r requirements.txt
Run the Flask API:
python app.py
Open another terminal and start the frontend:
cd frontend
npm install
npm start
Open Jupyter Notebook and train the model:
jupyter notebook phishing_detection.ipynb
How to Load Models
Load the Machine Learning Model (Python)
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the pre-trained model
with open('models/phishing_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict using the model
sample_data = [[0,1,1,0,1,-1,0,1,-1,1]]  # Example input
prediction = model.predict(sample_data)
print("Prediction:", "Phishing" if prediction[0] == -1 else "Legitimate")
Load Flask API (Python Backend)
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("models/phishing_model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
Load JavaScript Frontend (React.js)
import React, { useState } from 'react';

function App() {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: extractFeatures(url) })
    });
    const data = await response.json();
    setResult(data.prediction === -1 ? 'Phishing' : 'Legitimate');
  };

  return (
    <div>
      <h1>Phishing Detection System</h1>
      <input type="text" placeholder="Enter URL" onChange={(e) => setUrl(e.target.value)} />
      <button onClick={handleSubmit}>Check</button>
      {result && <h2>Result: {result}</h2>}
    </div>
  );
}
export default App;
Key Code Components
Feature Extraction: Extracts URL-based features for classification.
Hybrid Machine Learning Model: Combines Logistic Regression, SVM, and Decision Tree.
Flask API: Backend service for prediction.
React Frontend: User-friendly interface for checking URLs.
Summary
✔ Step 1: Install dependencies
✔ Step 2: Load machine learning models
✔ Step 3: Run Flask API & React Frontend
✔ Step 4: Use Jupyter Notebook for model training
✔ Step 5: Predict phishing websites with high accuracy

External Description
The Phishing Detection System leverages Hybrid Machine Learning to analyze URL features and detect malicious websites. This project aims to protect users from phishing attacks by providing a reliable, real-time detection system. It combines Decision Trees, SVM, and Logistic Regression for high-accuracy classification.

Contributing
Feel free to contribute by submitting issues or pull requests!

License
This project is licensed under the MIT License.

Acknowledgments
UCI Machine Learning Repository for dataset.
Flask & React for web-based implementation.
Scikit-learn for machine learning models.
