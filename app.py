from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load(os.path.join('model', 'model.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect features from form and convert to float
        features = [float(x) for x in request.form.values()]
        
        # Model expects a 2D array
        prediction = model.predict([features])
        
        return render_template('index.html', prediction_text=f"🌧️ Rainfall Prediction: {prediction[0]}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
