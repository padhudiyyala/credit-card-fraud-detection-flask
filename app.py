import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def train_dummy_model():
    """
    Trains an XGBoost model. 
    If 'creditcard.csv' exists in the project folder, it loads real data.
    Otherwise, it generates synthetic fraud data for immediate testing.
    """
    global model, scaler
    print("Initializing Machine Learning Model...")
    
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

    if os.path.exists('creditcard.csv'):
        print("Found creditcard.csv! Loading dataset...")
        df = pd.read_csv('creditcard.csv')
        X = df[feature_names].copy()
        y = df['Class']
    else:
        print("No dataset file found. Generating sample data for demo...")
        np.random.seed(42)
        n_samples = 5000
        
        # Synthetic V1-V28 features + Amount
        X_dummy = np.random.randn(n_samples, 29)
        # ~2% Fraud rate
        y_dummy = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02]) 
        
        X = pd.DataFrame(X_dummy, columns=feature_names)
        y = pd.Series(y_dummy, name='Class')

    # Fit scaler on Amount with feature names preserved
    scaler = StandardScaler()
    X[['Amount']] = scaler.fit_transform(X[['Amount']])

    # Train Model
    model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    print("✅ Model trained successfully and ready for web requests!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        amount = float(data.get('amount', 0))
        
        # 1. Scale amount using DataFrame (preserves feature name expected by scaler)
        amount_df = pd.DataFrame([[amount]], columns=['Amount'])
        scaled_amount = scaler.transform(amount_df)[0][0]
        
        # 2. Reconstruct all 29 features with exact column names expected by XGBoost
        v_features = [float(data.get(f'v{i}', 0.0)) for i in range(1, 29)]
        all_features = v_features + [scaled_amount]
        
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_df = pd.DataFrame([all_features], columns=feature_names)
        
        # 3. Model Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        is_fraud = bool(prediction == 1 or probability > 50.0)

        return jsonify({
            'status': 'success',
            'is_fraud': is_fraud,
            'risk_score': round(float(probability), 2),
            'message': 'CRITICAL WARNING: High Risk Fraud Detected!' if is_fraud else 'Legitimate Transaction Approved'
        })
    except Exception as e:
        print(f"Prediction Error: {e}")  # Outputs error in terminal window for debugging
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    train_dummy_model()
    app.run(debug=True, port=5000)