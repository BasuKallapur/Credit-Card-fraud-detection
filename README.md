# Credit Card Fraud Detection

This project implements machine learning models for credit card fraud detection, with a focus on identifying and addressing data leakage issues while providing a practical application.

## Project Overview

The project consists of:

1. **Analysis Script** (`credit_card_fraud_detection.py`): Compares four models (Logistic Regression, Decision Tree, SGD, Random Forest)
2. **Model Training** (`fraud_predictor.py`): Trains and saves the best model (Random Forest)
3. **Prediction Interface** (`predict_fraud.py`): User-friendly interface for fraud prediction

## Model Comparison

The `credit_card_fraud_detection.py` script performs a comprehensive analysis of four different machine learning models:

1. **Logistic Regression**
   - Configuration: Strong regularization (C=0.01, 0.1), balanced class weights
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix

2. **Decision Tree**
   - Configuration: Limited max_depth (3-4), high min_samples_split (50-100), high min_samples_leaf (20-50)
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix

3. **SGD Classifier** (Linear SVM)
   - Configuration: L2 penalty, hinge loss, strong regularization
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix

4. **Random Forest** (Best performing model)
   - Configuration: Limited tree depth, high minimum samples for splits/leaves, balanced class weights
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix
   - Feature importance analysis to identify key predictors

The script generates various visualization files including:
- Confusion matrices for each model
- ROC curves showing model performance
- Feature importance chart (for Random Forest)
- Precision-recall curves
- Model comparison summary

## Key Findings: Data Leakage Issue

During our analysis, we discovered an important issue that's common in machine learning:

- All models achieved suspiciously high accuracy (99.98% for Random Forest)
- Initially, we suspected overfitting and implemented anti-overfitting measures:
  - Reduced tree depth
  - Increased minimum samples for splits/leaves
  - Added regularization
  - Used balanced class weights
- Despite these measures, accuracy remained near-perfect

### The Real Issue: Data Leakage

Further investigation revealed this wasn't traditional overfitting but **data leakage**:

- The `ratio_to_median_purchase_price` feature had a 46% correlation with fraud
- When we removed just this one feature, accuracy dropped dramatically from 99.98% to 48.16%
- This indicated that a single feature was essentially "giving away" the answer

This is an important finding because:
1. In real-world fraud detection, such a powerful single indicator might not be available in real-time
2. Fraudsters could potentially learn to circumvent this single detection mechanism
3. The model wasn't learning complex patterns but relying heavily on one feature

For this project, we decided to keep using all features since:
1. It demonstrates the importance of feature analysis in fraud detection
2. The model still performs correctly according to the available data
3. It highlights how important it is to understand your features before deploying models

## Dataset

The dataset contains the following features:
- `distance_from_home` - Distance from home where the transaction happened
- `distance_from_last_transaction` - Distance from last transaction
- `ratio_to_median_purchase_price` - Ratio of purchased price to median purchase price (our critical feature)
- `repeat_retailer` - Is the transaction from same retailer (1 for yes, 0 for no)
- `used_chip` - Is the transaction through chip (1 for yes, 0 for no)
- `used_pin_number` - Is the transaction using PIN number (1 for yes, 0 for no)
- `online_order` - Is the transaction an online order (1 for yes, 0 for no)
- `fraud` - Is the transaction fraudulent (target variable)

## Setup

1. Ensure you have Python installed (3.7+ recommended)
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run the Full Analysis

For a comprehensive analysis of different models and to see the data leakage issue:

```
python credit_card_fraud_detection.py
```

This generates visualizations including:
- Confusion matrices
- ROC curves
- Feature importance (showing the dominance of ratio_to_median_purchase_price)
- Model comparison charts

### Option 2: Train and Save the Model

To train and save the optimal Random Forest model:

```
python fraud_predictor.py
```

This creates:
- `fraud_model.pkl` - The trained model
- `scaler.pkl` - The standardization scaler

### Option 3: Predict Fraud on New Transactions

For an interactive prediction interface:

```
python predict_fraud.py
```

This allows you to:
- Enter transaction details
- Get instant fraud predictions
- See explanations of risk factors

## Implementation Details

The Random Forest model is configured with:
- Reduced tree depth (max_depth=5)
- Higher min_samples_split (50) and min_samples_leaf (20)
- Balanced class weights
- Square root feature selection

These parameters were chosen to minimize overfitting, even though the main issue turned out to be data leakage rather than traditional overfitting.

## Module Structure

- `fraud_predictor.py` - Core model training and prediction functionality
- `predict_fraud.py` - User interface for making predictions
- `credit_card_fraud_detection.py` - Comprehensive analysis and model comparison

## Learning Outcomes

This project demonstrates:
1. How to implement machine learning for fraud detection
2. The importance of thorough feature analysis
3. How to identify and understand data leakage issues
4. Building a practical, user-friendly prediction interface