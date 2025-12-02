### Car Demo Video
[Click to watch the demo video]
https://drive.google.com/file/d/1ujrhxTRMR8GvlOdFIlEkDMkYWdHY3_tZ/view?usp=sharing

Wholesale Customer Analysis & Prediction System
<div align="center">
https://img.shields.io/badge/Python-3.7+-blue.svg
https://img.shields.io/badge/Flask-2.0+-green.svg
https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg
https://img.shields.io/badge/License-MIT-yellow.svg

A Machine Learning Web Application for Customer Channel Prediction and Segmentation

https://img.shields.io/badge/Live-Demo-red.svg
https://img.shields.io/badge/%F0%9F%93%96-Documentation-blue.svg
https://img.shields.io/badge/%F0%9F%90%9B-Report%2520Bug-red.svg

</div>
📋 Table of Contents
✨ Features

🚀 Quick Start

📊 Project Overview

🏗️ Architecture

🔧 Installation

📖 Usage Guide

📁 Project Structure

🤖 Machine Learning

🌐 Web Interface

📈 Results

🚀 Deployment

🤝 Contributing

📄 License

👨‍💻 Author

✨ Features
🎯 Core Features
Real-time Prediction: Predict customer channel (Retail/Horeca) based on spending patterns

Customer Segmentation: K-Means clustering to group similar customers

Interactive Dashboard: User-friendly web interface with multiple views

Data Visualization: Clean presentation of data and statistics

Model Management: Easy model updates and persistence

📊 Data Analysis
Complete dataset exploration and preprocessing

Statistical summary generation

Feature correlation analysis

Data quality assessment

🔮 Prediction Capabilities
Support Vector Machine (SVM) model with 87.9% accuracy

Input validation and error handling

Confidence scoring for predictions

Batch prediction support

🚀 Quick Start
Prerequisites
Python 3.7 or higher

pip package manager

One-Command Installation
bash
# Clone the repository
git clone https://github.com/yourusername/wholesale-project.git
cd wholesale-project

# Install dependencies (Windows/macOS/Linux)
pip install -r requirements.txt

# Run the application
python app.py

# Open browser and navigate to:
# http://localhost:5000
Requirements File
txt
Flask==2.3.3
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
📊 Project Overview
Business Problem
Wholesale distributors need to classify customers into Retail or Horeca (Hotel/Restaurant/Cafe) channels to optimize marketing, inventory, and customer relationship strategies.

Dataset
Source: UCI Machine Learning Repository

Records: 440 customers

Features: 8 attributes including product category spending

Target: Channel (1=Horeca, 2=Retail)

Key Statistics
Model Accuracy: 87.9%

Training Samples: 308

Testing Samples: 132

Features Used: 5

Clusters: 3 customer segments

🏗️ Architecture
Tech Stack
text
Frontend:   HTML5, CSS3, Jinja2 Templates
Backend:    Flask (Python)
ML Engine:  Scikit-learn, Pandas, NumPy
Database:   CSV Files
Model:      Pickle serialization
System Architecture











🔧 Installation
Step-by-Step Setup
1. Clone Repository
bash
git clone https://github.com/yourusername/wholesale-project.git
cd wholesale-project
2. Create Virtual Environment (Recommended)
bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Verify Installation
bash
python -c "import flask, pandas, sklearn; print('All packages installed successfully!')"
5. Run Application
bash
python app.py
6. Access Application
Open your browser and navigate to: http://localhost:5000

📖 Usage Guide
Web Interface Navigation
Home Page
URL: /

Description: Project overview and main navigation hub

Features: Quick links to all application sections

Make Predictions
URL: /predict

Description: Input form for customer spending data

Input Fields:

Milk Products Spending (€)

Grocery Spending (€)

Frozen Products Spending (€)

Detergents & Paper Spending (€)

Delicatessen Spending (€)

View Results
URL: /result

Description: Displays prediction results

Output: Predicted channel with confidence score

Explore Data
URL: /dataset

Description: View first 10 records of the dataset

Features: Interactive table with sorting

Statistical Analysis
URL: /summary

Description: Detailed statistical summary

Features: Count, mean, std, min/max, percentiles

Customer Segmentation
URL: /cluster

Description: K-Means clustering results

Features: Customer groups and spending patterns

API Usage
python
import requests
import json

# Sample API call for prediction
data = {
    "milk": 5000,
    "grocery": 8000,
    "frozen": 2000,
    "detergents_paper": 1000,
    "delicassen": 500
}

response = requests.post("http://localhost:5000/predict", json=data)
result = response.json()
print(f"Predicted Channel: {result['prediction']}")
📁 Project Structure
text
wholesale-customer-analysis/
│
├── 📁 static/
│   └── style.css          # CSS stylesheet
│
├── 📁 templates/
│   ├── home.html          # Home page
│   ├── predict.html       # Prediction form
│   ├── result.html        # Results page
│   ├── dataset.html       # Dataset viewer
│   ├── summary.html       # Statistical summary
│   └── cluster.html       # Clustering results
│
├── 📁 notebooks/
│   ├── lab 9.ipynb       # Data exploration
│   ├── lab 10.ipynb      # Data preprocessing
│   └── lab 11.ipynb      # Model training
│
├── 📁 models/
│   └── model_svc.pkl     # Trained ML model
│
├── 📁 data/
│   ├── Wholesale customers data.csv    # Original dataset
│   └── processed_data.csv             # Cleaned dataset
│
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore file
🤖 Machine Learning
Model Details
Algorithm: Support Vector Classifier (SVC)

Accuracy: 87.9%

Features Used: 5

Training Time: < 2 seconds

Model Size: ~10 KB

Training Process
python
# Simplified training code
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load and prepare data
X = df[['Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
y = df['Channel']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)  # 0.879
Clustering
Algorithm: K-Means

Clusters: 3

Purpose: Customer segmentation

Features: All product categories

🌐 Web Interface
Screenshots
Home Page
text
┌─────────────────────────────────────────────┐
│   🏪 Wholesale Customer Analysis            │
│   Predict Customer Channel & Analyze        │
│                                             │
│   [🔮 Predict Channel]  [📊 View Dataset]   │
│   [📈 Data Summary]     [🎯 Clustering]     │
└─────────────────────────────────────────────┘
Prediction Form
text
┌─────────────────────────────────────────────┐
│   🔮 Predict Customer Channel               │
│                                             │
│   Milk Products:      [_______] €          │
│   Grocery:            [_______] €          │
│   Frozen Products:    [_______] €          │
│   Detergents & Paper: [_______] €          │
│   Delicatessen:       [_______] €          │
│                                             │
│            [Predict Channel]                │
└─────────────────────────────────────────────┘
Responsive Design
Mobile-friendly layout

Cross-browser compatible

Fast loading times

Accessible interface

📈 Results
Model Performance
text
Model Accuracy: 87.9%
Precision: 0.88
Recall: 0.87
F1-Score: 0.87
Confusion Matrix
text
Actual/Predicted   Retail   Horeca
──────────────────────────────────
Retail              85       12
Horeca              4        31
Feature Importance
Grocery - 32% importance

Milk - 28% importance

Detergents_Paper - 22% importance

Frozen - 12% importance

Delicassen - 6% importance
