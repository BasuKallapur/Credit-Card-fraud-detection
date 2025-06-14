# Credit Card Fraud Detection
Demo Video: [Watch Demo](https://drive.google.com/file/d/1lZ4IcC6M86f20kwWBCAvDLqeP0-Wv0Xa/view?usp=drive_link)

## Project Overview

This project implements machine learning models for credit card fraud detection, with a focus on identifying and addressing data leakage issues while providing a practical application through both CLI and web interfaces.

### Key Components

1. **Analysis Script** (`credit_card_fraud_detection.py`): Compares four models (Logistic Regression, Decision Tree, SGD, Random Forest)
2. **Model Training** (`fraud_predictor.py`): Trains and saves the best model (Random Forest)
3. **Prediction Interfaces**:
   - Command-line interface (`predict_fraud.py`): Simple CLI for fraud prediction
   - Streamlit dashboard (`app.py`): Interactive web application with visualizations

## Features

- **Fraud Prediction**: Input transaction details and get real-time fraud predictions
- **Data Exploration**: Analyze the dataset with interactive visualizations
- **Model Performance**: Compare the performance of different machine learning models
- **Interactive Visualizations**: Explore data relationships with dynamic charts
- **Educational Value**: Demonstrates data leakage issues in machine learning

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

## Model Comparison

The project compares four different machine learning models:

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

## Setup and Installation

### Prerequisites

- Python 3.7 or higher (Python 3.10.12 recommended)
- pip package manager

### Installation Steps

1. Clone the repository:
```
git clone <repository-url>
cd Credit-Card-fraud-detection
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Ensure the model is trained and saved:
```
python fraud_predictor.py
```
This will create the necessary model files (`fraud_model.pkl` and `scaler.pkl`).

## Usage

### Option 1: Run the Streamlit Web Application

For an interactive web dashboard with visualizations and predictions:

```
streamlit run app.py
```

Access the dashboard in your web browser at: http://localhost:8501

The dashboard includes:
- Fraud prediction interface
- Data exploration visualizations
- Model performance comparisons
- Feature importance analysis

### Option 2: Run the Full Analysis

For a comprehensive analysis of different models and to see the data leakage issue:

```
python credit_card_fraud_detection.py
```

This generates visualizations including:
- Confusion matrices
- ROC curves
- Feature importance (showing the dominance of ratio_to_median_purchase_price)
- Model comparison charts

### Option 3: Use the Command-Line Prediction Interface

For a simple command-line prediction interface:

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

## Deployment

The project includes configuration files for deployment to:
- Heroku (Procfile)
- Railway (railway.json)

To deploy to Railway:
1. Push your code to a GitHub repository
2. Connect your repository to Railway
3. Railway will automatically deploy using the configuration in railway.json

## Project Structure

```
Credit-Card-fraud-detection/
├── app.py                        # Streamlit web application
├── card_transdata_sample.csv     # Sample dataset (1000 records)
├── create_sample.py              # Script to create sample dataset
├── credit_card_fraud_detection.py # Main analysis script
├── fraud_model.pkl               # Trained model file
├── fraud_predictor.py            # Model training and prediction module
├── guide.md                      # Presentation guide
├── model_metrics.csv             # Model performance metrics
├── predict_fraud.py              # CLI prediction interface
├── Procfile                      # Heroku deployment configuration
├── railway.json                  # Railway deployment configuration
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python runtime specification
├── scaler.pkl                    # Feature scaling model
├── static/                       # Static assets for web app
└── .streamlit/                   # Streamlit configuration
```

## Learning Outcomes

This project demonstrates:
1. How to implement machine learning for fraud detection
2. The importance of thorough feature analysis
3. How to identify and understand data leakage issues
4. Building practical, user-friendly prediction interfaces (both CLI and web)
5. Deploying machine learning applications to cloud platforms

## Troubleshooting

- **Model files not found**: Run `python fraud_predictor.py` to generate the model files
- **Dataset not found**: Ensure the dataset CSV file is in the project directory
- **Visualization errors**: Make sure all required libraries are installed
- **Deployment issues**: Check the logs for specific error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
