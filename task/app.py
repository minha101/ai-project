
from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Prevent caching issues

# Load dataset
df = pd.read_csv("Wholesale customers data.csv")

# Load trained model
try:
    model_svc = pickle.load(open('model_svc.pkl', 'rb'))
except:
    model_svc = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        try:
            # Get form data
            milk = float(request.form['milk'])
            grocery = float(request.form['grocery'])
            frozen = float(request.form['frozen'])
            detergents_paper = float(request.form['detergents_paper'])
            delicassen = float(request.form['delicassen'])
            
            # Create input array
            input_data = np.array([[milk, grocery, frozen, detergents_paper, delicassen]])
            
            # Make prediction
            if model_svc:
                prediction = model_svc.predict(input_data)[0]
                channel_name = "Horeca (Hotel/Restaurant/Cafe)" if prediction == 1 else "Retail"
                
                return render_template("result.html", 
                                     prediction=channel_name,
                                     milk=milk,
                                     grocery=grocery,
                                     frozen=frozen,
                                     detergents_paper=detergents_paper,
                                     delicassen=delicassen)
            else:
                return render_template("result.html", error="Model not available. Please check if model file exists.")
        
        except Exception as e:
            return render_template("result.html", error=f"Error in prediction: {str(e)}")
    
    return render_template("predict.html")

@app.route("/dataset")
def dataset():
    table = df.head(10).to_html(classes="table table-bordered", index=False)
    return render_template("dataset.html", table=table)

@app.route("/summary")
def summary():
    summary_table = df.describe().to_html(classes="table table-bordered")
    return render_template("summary.html", table=summary_table)

@app.route("/cluster")
def cluster():
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cluster = df.copy()
    cluster_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    df_cluster["Cluster"] = kmeans.fit_predict(df_cluster[cluster_features])
    
    cluster_table = df_cluster[['Channel', 'Region', 'Cluster'] + cluster_features].head(15).to_html(
        classes="table table-bordered", index=False)
    return render_template("cluster.html", table=cluster_table)

# Add this to force refresh CSS
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path, endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == "__main__":

    app.run(debug=True)

