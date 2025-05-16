# Credit Card Fraud Detection

This project implements and compares four classification models for credit card fraud detection:
1. Logistic Regression
2. Decision Tree Classifier
3. Support Vector Classifier (SVC)
4. Random Forest Classifier

## Dataset

The dataset contains the following features:
- `distance_from_home` - The distance from home where the transaction happened
- `distance_from_last_transaction` - The distance from last transaction happened
- `ratio_to_median_purchase_price` - Ratio of purchased price transaction to median purchase price
- `repeat_retailer` - Is the transaction happened from same retailer
- `used_chip` - Is the transaction through chip (credit card)
- `used_pin_number` - Is the transaction happened by using PIN number
- `online_order` - Is the transaction an online order
- `fraud` - Is the transaction fraudulent (target variable)

## Setup

1. Ensure you have Python installed (3.7+ recommended)
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. The script is currently set to use the dataset at `C:\Users\basuk\Downloads\bassu\card_transdata.csv`
   (modify the file path in the script if needed)

## Running the Analysis

Run the script with:
```
python credit_card_fraud_detection.py
```

## Outputs

The script will generate:
1. A detailed classification report for each model
2. Confusion matrices saved as PNG files
3. ROC curves for each model
4. A feature importance plot for Random Forest
5. Model accuracy comparison chart
6. Precision-Recall curve for all models

## Hyperparameter Tuning

The script performs GridSearchCV for each model to find the optimal parameters.
- Logistic Regression: Tests regularization strength, solver, and class weights
- Decision Tree: Tests max depth, min samples split/leaf, and class weights
- SVC: Tests C parameter, kernel type, gamma, and class weights 
- Random Forest: Tests number of estimators, max depth, min samples split/leaf, and class weights 