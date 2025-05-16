"""
Credit Card Fraud Detection - Model Training and Prediction

This module provides functionality to train a machine learning model for credit card fraud detection
and make predictions on new transaction data.

Key components:
- train_save_model(): Trains a Random Forest model on credit card transaction data and saves it
- predict_transaction(): Predicts whether a transaction is fraudulent based on its features

The model uses the following features:
- distance_from_home: Distance from home where transaction occurred
- distance_from_last_transaction: Distance from last transaction
- ratio_to_median_purchase_price: Ratio of purchase price to median
- repeat_retailer: Whether transaction occurred at a repeat retailer (0 or 1)
- used_chip: Whether chip was used (0 or 1) 
- used_pin_number: Whether PIN was used (0 or 1)
- online_order: Whether it was an online order (0 or 1)

Usage:
1. Run this file directly to train and save the model
2. Import predict_transaction() in other scripts to make predictions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_save_model():
    """Train the fraud detection model and save it for future use"""
    print("Training and saving model...")
    
    # Load the dataset
    try:
        print("Attempting to load card_transdata copy.csv")
        df = pd.read_csv("card_transdata copy.csv")
        print("Loaded card_transdata copy.csv")
    except Exception as e1:
        print(f"Error loading card_transdata copy.csv: {e1}")
        try:
            print("Attempting to load card_transdata.csv")
            df = pd.read_csv("card_transdata.csv")
            print("Loaded card_transdata.csv")
        except Exception as e2:
            print(f"Error loading card_transdata.csv: {e2}")
            raise
    
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Prepare the data
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Create and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a Random Forest model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    print("Training model...")
    # Fit the model
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'fraud_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model trained and saved successfully!")
    return model, scaler

def predict_transaction(distance_from_home, distance_from_last_transaction, 
                       ratio_to_median_purchase_price, repeat_retailer, 
                       used_chip, used_pin_number, online_order,
                       model=None, scaler=None):
    """
    Predict if a transaction is fraudulent based on its features
    
    Parameters:
    - distance_from_home: Distance from home where transaction occurred
    - distance_from_last_transaction: Distance from last transaction
    - ratio_to_median_purchase_price: Ratio of purchase price to median
    - repeat_retailer: Whether transaction occurred at a repeat retailer (0 or 1)
    - used_chip: Whether chip was used (0 or 1)
    - used_pin_number: Whether PIN was used (0 or 1)
    - online_order: Whether it was an online order (0 or 1)
    - model: Optional pre-loaded model (will load from disk if None)
    - scaler: Optional pre-loaded scaler (will load from disk if None)
    
    Returns:
    - prediction: 0 for legitimate, 1 for fraudulent
    - probability: Probability of fraud (0-1)
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        # Check if model and scaler exist
        if not os.path.exists('fraud_model.pkl') or not os.path.exists('scaler.pkl'):
            model, scaler = train_save_model()
        else:
            model = joblib.load('fraud_model.pkl')
            scaler = joblib.load('scaler.pkl')
    
    # Create transaction data
    transaction = pd.DataFrame({
        'distance_from_home': [distance_from_home],
        'distance_from_last_transaction': [distance_from_last_transaction],
        'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
        'repeat_retailer': [repeat_retailer],
        'used_chip': [used_chip],
        'used_pin_number': [used_pin_number],
        'online_order': [online_order]
    })
    
    # Scale and predict
    scaled_transaction = scaler.transform(transaction)
    prediction = model.predict(scaled_transaction)[0]
    probability = model.predict_proba(scaled_transaction)[0][1]
    
    return prediction, probability

# Simple execution to train model if run directly
if __name__ == "__main__":
    if not os.path.exists('fraud_model.pkl') or not os.path.exists('scaler.pkl'):
        train_save_model()
    else:
        print("Model already exists. To retrain, delete fraud_model.pkl and scaler.pkl files.") 