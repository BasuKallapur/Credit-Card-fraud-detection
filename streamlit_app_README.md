# Credit Card Fraud Detection Dashboard

This Streamlit dashboard provides an interactive interface for exploring credit card fraud detection models and making predictions on transaction data.

## Features

- **Fraud Prediction**: Input transaction details and get real-time fraud predictions
- **Data Exploration**: Analyze the dataset with interactive visualizations
- **Model Performance**: Compare the performance of different machine learning models
- **Interactive Visualizations**: Explore data relationships with dynamic charts

## Screenshots

[Screenshot images would be here]

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
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

4. Run the Streamlit app:
```
streamlit run app.py
```

5. Access the dashboard in your web browser at: http://localhost:8501

## Usage Guide

### Fraud Prediction

1. Navigate to the "Fraud Prediction" page using the sidebar.
2. Input transaction details:
   - Distance from home (miles)
   - Distance from last transaction (miles)
   - Ratio to median purchase price
   - Whether it's a repeat retailer
   - Whether chip was used
   - Whether PIN was used
   - Whether it was an online order
3. Click the "Predict" button to get a fraud prediction.
4. Review the results, which include:
   - Prediction (Fraud/Legitimate)
   - Fraud probability gauge
   - Risk factor analysis
   - Feature contribution visualization

### Data Exploration

1. Navigate to the "Data Exploration" page.
2. Use the tabs to explore different aspects of the data:
   - Distribution: View fraud distribution and feature histograms
   - Correlation: Explore relationships between features
   - Feature Importance: See which features contribute most to predictions
   - Statistics: Review summary statistics for the dataset

### Model Performance

1. Navigate to the "Model Performance" page.
2. Use the tabs to examine details for each model:
   - Random Forest (Best performing model)
   - Logistic Regression
   - Decision Tree
   - SGD Classifier
3. Review model configurations, metrics, and visualizations
4. Compare all models using the comparison chart and table

## Additional Resources

- Full project analysis: `credit_card_fraud_detection.py`
- Model training code: `fraud_predictor.py`
- CLI prediction interface: `predict_fraud.py`

## Troubleshooting

- **Model files not found**: Run `python fraud_predictor.py` to generate the model files
- **Dataset not found**: Ensure the dataset CSV file is in the project directory
- **Visualization errors**: Make sure all required libraries are installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 