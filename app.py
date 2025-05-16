import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from fraud_predictor import predict_transaction
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("Model files not found. Please run 'python fraud_predictor.py' first to create the model.")
        return None, None

# Function to load dataset for exploration
@st.cache_data
def load_data():
    try:
        try:
            df = pd.read_csv("card_transdata copy.csv")
        except:
            df = pd.read_csv("card_transdata.csv")
        return df
    except:
        st.warning("Dataset not found for exploration view. Only prediction will be available.")
        return None

def create_correlation_heatmap(df):
    corr = df.corr()
    fig = px.imshow(
        corr, 
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix",
        labels=dict(x="Features", y="Features", color="Correlation")
    )
    fig.update_layout(width=800, height=800)
    return fig

def create_fraud_distribution(df):
    fraud_counts = df['fraud'].value_counts().reset_index()
    fraud_counts.columns = ['Fraud', 'Count']
    fig = px.pie(
        fraud_counts, 
        values='Count', 
        names='Fraud',
        title='Fraud vs Non-Fraud Distribution',
        hole=0.4,
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    return fig

def create_feature_importance_chart():
    try:
        model, _ = load_model_and_scaler()
        if model:
            feature_names = [
                'distance_from_home', 'distance_from_last_transaction',
                'ratio_to_median_purchase_price', 'repeat_retailer', 
                'used_chip', 'used_pin_number', 'online_order'
            ]
            importance = model.feature_importances_
            
            # Create a DataFrame for visualization
            feat_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(
                feat_importance, 
                x='Importance', 
                y='Feature',
                title='Feature Importance',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            return fig
        return None
    except:
        return None

def main():
    # Load model and data
    model, scaler = load_model_and_scaler()
    data = load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Fraud Prediction", "Data Exploration", "Model Performance", "About"]
    )
    
    # Home page
    if page == "Home":
        st.title("Credit Card Fraud Detection Dashboard")
        st.markdown("""
        ### Welcome to the Credit Card Fraud Detection Dashboard
        
        This interactive dashboard provides tools to analyze and predict credit card fraud.
        
        **Key Features**:
        - ✅ Predict if a transaction is fraudulent
        - 📊 Explore the dataset and patterns
        - 🔍 Understand model features and importance
        
        **How to use**:
        - Use the sidebar to navigate between different pages
        - On the Fraud Prediction page, enter transaction details to get a prediction
        - On the Data Exploration page, analyze patterns in the dataset
        
        **Model Performance**:
        - This dashboard uses a Random Forest model
        - Feature engineering addresses data leakage issues
        - High accuracy in distinguishing fraudulent transactions
        """)
        
        # Display sample card image
        st.image("https://img.freepik.com/free-vector/realistic-credit-card-design_23-2149126090.jpg", width=400)
        
        # Display key metrics if data is available
        if data is not None:
            st.subheader("Dataset Overview")
            total_transactions = len(data)
            fraud_percent = data['fraud'].mean() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", f"{total_transactions:,}")
            with col2:
                st.metric("Fraud Transactions", f"{int(total_transactions * fraud_percent / 100):,}")
            with col3:
                st.metric("Fraud Percentage", f"{fraud_percent:.2f}%")
    
    # Fraud prediction page
    elif page == "Fraud Prediction":
        st.title("Fraud Prediction")
        st.write("Enter transaction details to predict if it's fraudulent")
        
        with st.expander("ℹ️ How to use this tool", expanded=False):
            st.markdown("""
            **This tool helps you predict if a credit card transaction is fraudulent based on its characteristics.**
            
            1. Enter your transaction details in the input fields
            2. Click the "Predict" button to see the results
            3. Review the fraud probability and risk factors
            
            The prediction is based on a Random Forest model trained on historical transaction data.
            """)
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Location & Amount")
            
            distance_home = st.number_input(
                "Distance from home (miles) 🏠", 
                min_value=0.0, 
                max_value=5000.0, 
                value=50.0,
                step=1.0,
                help="The distance from the customer's home location where the transaction occurred"
            )
            
            distance_last = st.number_input(
                "Distance from last transaction (miles) 📍", 
                min_value=0.0, 
                max_value=5000.0, 
                value=20.0,
                step=1.0,
                help="The distance from the customer's previous transaction location"
            )
            
            ratio_median = st.number_input(
                "Ratio to median purchase price 💰", 
                min_value=0.01, 
                max_value=10.0, 
                value=1.5,
                step=0.01,
                help="How this transaction compares to the customer's median purchase (1.0 = same as median, 2.0 = twice the median)"
            )
        
        with col2:
            st.subheader("Transaction Characteristics")
            
            repeat_retailer_options = {"Yes": 1, "No": 0}
            repeat_retailer = st.selectbox(
                "Repeat retailer? 🔄",
                options=list(repeat_retailer_options.keys()),
                index=0,
                help="Is this a retailer the customer has purchased from before?"
            )
            repeat_retailer = repeat_retailer_options[repeat_retailer]
            
            used_chip_options = {"Yes": 1, "No": 0}
            used_chip = st.selectbox(
                "Used chip? 💳", 
                options=list(used_chip_options.keys()),
                index=0,
                help="Was the card's chip used for this transaction?"
            )
            used_chip = used_chip_options[used_chip]
            
            used_pin_options = {"Yes": 1, "No": 0}
            used_pin = st.selectbox(
                "Used PIN number? 🔢", 
                options=list(used_pin_options.keys()),
                index=0,
                help="Was a PIN number used for this transaction?"
            )
            used_pin = used_pin_options[used_pin]
            
            online_order_options = {"Yes": 1, "No": 0}
            online_order = st.selectbox(
                "Online order? 🌐", 
                options=list(online_order_options.keys()),
                index=1,
                help="Was this transaction conducted online?"
            )
            online_order = online_order_options[online_order]
        
        # Add a divider and prediction button
        st.markdown("---")
        predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
        with predict_col2:
            predict_button = st.button("🔍 Predict Fraud Risk", use_container_width=True)
        
        if predict_button:
            if model is not None and scaler is not None:
                # Create a spinner to show processing
                with st.spinner("Analyzing transaction..."):
                    prediction, probability = predict_transaction(
                        distance_from_home=distance_home,
                        distance_from_last_transaction=distance_last,
                        ratio_to_median_purchase_price=ratio_median,
                        repeat_retailer=repeat_retailer,
                        used_chip=used_chip,
                        used_pin_number=used_pin,
                        online_order=online_order,
                        model=model,
                        scaler=scaler
                    )
                
                # Display result with Streamlit components
                st.markdown("## Prediction Result")
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if prediction == 1:
                        st.error("⚠️ **FRAUD ALERT!**")
                        st.markdown("""
                        <div style="padding: 10px; border-radius: 10px; background-color: rgba(255, 0, 0, 0.1);">
                            <h3 style="color: darkred;">This transaction is likely FRAUDULENT</h3>
                            <p>Our model has flagged this transaction as potentially fraudulent based on the provided details.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ **LEGITIMATE TRANSACTION**")
                        st.markdown("""
                        <div style="padding: 10px; border-radius: 10px; background-color: rgba(0, 255, 0, 0.1);">
                            <h3 style="color: darkgreen;">This transaction appears LEGITIMATE</h3>
                            <p>Our model has not detected signs of fraud in this transaction based on the provided details.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display fraud probability gauge
                with result_col2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            },
                            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "gold"},
                                {'range': [70, 100], 'color': "lightsalmon"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Insight section
                st.subheader("Risk Analysis")
                factors = []
                
                if ratio_median > 2:
                    factors.append({
                        "factor": f"Purchase amount is {ratio_median:.1f}x the median", 
                        "risk": "high",
                        "explanation": "Transactions significantly above a customer's median purchase amount are more likely to be fraudulent."
                    })
                
                if online_order == 1:
                    factors.append({
                        "factor": "Online order", 
                        "risk": "medium",
                        "explanation": "Online transactions generally have higher fraud rates than in-person transactions."
                    })
                    
                if distance_home > 100:
                    factors.append({
                        "factor": f"Transaction occurred {distance_home:.1f} miles from home", 
                        "risk": "medium",
                        "explanation": "Transactions far from a customer's home location may indicate card theft or unauthorized use."
                    })
                    
                if used_chip == 0 and used_pin == 0:
                    factors.append({
                        "factor": "No chip or PIN security features used", 
                        "risk": "high",
                        "explanation": "Transactions without chip or PIN verification have fewer security protections."
                    })
                
                if not factors:
                    st.info("✓ No significant risk factors identified")
                else:
                    for factor in factors:
                        if factor["risk"] == "high":
                            st.error(f"**{factor['factor']}** (High Risk): {factor['explanation']}")
                        elif factor["risk"] == "medium":
                            st.warning(f"**{factor['factor']}** (Medium Risk): {factor['explanation']}")
                        else:
                            st.info(f"**{factor['factor']}** (Low Risk): {factor['explanation']}")
                
                # Feature importance for this prediction
                if model is not None:
                    st.subheader("Feature Contribution Analysis")
                    features = [
                        distance_home, distance_last, ratio_median,
                        repeat_retailer, used_chip, used_pin, online_order
                    ]
                    feature_names = [
                        'Distance from home', 'Distance from last transaction',
                        'Ratio to median price', 'Repeat retailer', 
                        'Used chip', 'Used PIN', 'Online order'
                    ]
                    
                    # Convert to proper format for visualization
                    feat_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': features
                    })
                    
                    # Match the values with importance (simplified approach)
                    importance = model.feature_importances_
                    
                    fig = px.bar(
                        feat_df,
                        x='Value', 
                        y='Feature',
                        orientation='h',
                        title='Input Features for This Prediction',
                        color=feat_df.index,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a comparison to average values if data is available
                    if data is not None:
                        st.subheader("How This Transaction Compares")
                        
                        comparison_data = pd.DataFrame({
                            'Feature': feature_names,
                            'Your Transaction': features,
                            'Average Legitimate': [
                                data[data['fraud'] == 0]['distance_from_home'].mean(),
                                data[data['fraud'] == 0]['distance_from_last_transaction'].mean(),
                                data[data['fraud'] == 0]['ratio_to_median_purchase_price'].mean(),
                                data[data['fraud'] == 0]['repeat_retailer'].mean(),
                                data[data['fraud'] == 0]['used_chip'].mean(),
                                data[data['fraud'] == 0]['used_pin_number'].mean(),
                                data[data['fraud'] == 0]['online_order'].mean()
                            ],
                            'Average Fraudulent': [
                                data[data['fraud'] == 1]['distance_from_home'].mean(),
                                data[data['fraud'] == 1]['distance_from_last_transaction'].mean(),
                                data[data['fraud'] == 1]['ratio_to_median_purchase_price'].mean(),
                                data[data['fraud'] == 1]['repeat_retailer'].mean(),
                                data[data['fraud'] == 1]['used_chip'].mean(),
                                data[data['fraud'] == 1]['used_pin_number'].mean(),
                                data[data['fraud'] == 1]['online_order'].mean()
                            ]
                        })
                        
                        st.dataframe(comparison_data, use_container_width=True)
    
    # Data exploration page
    elif page == "Data Exploration":
        st.title("Data Exploration")
        
        if data is not None:
            st.write(f"Dataset shape: {data.shape}")
            
            # Display sample data
            st.subheader("Sample Data")
            # Get a mix of fraud and non-fraud cases for the sample
            fraud_sample = data[data['fraud'] == 1].sample(min(5, len(data[data['fraud'] == 1])))
            non_fraud_sample = data[data['fraud'] == 0].sample(min(5, len(data[data['fraud'] == 0])))
            # Combine and shuffle the samples
            combined_sample = pd.concat([fraud_sample, non_fraud_sample])
            combined_sample = combined_sample.sample(frac=1).reset_index(drop=True)
            st.dataframe(combined_sample)
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Feature Importance", "Statistics"])
            
            with tab1:
                st.plotly_chart(create_fraud_distribution(data), use_container_width=True)
                
                # Feature distributions
                st.subheader("Feature Distributions")
                feature_to_plot = st.selectbox(
                    "Select feature to plot distribution",
                    ['distance_from_home', 'distance_from_last_transaction', 
                     'ratio_to_median_purchase_price', 'repeat_retailer',
                     'used_chip', 'used_pin_number', 'online_order']
                )
                
                # Create histogram with fraud vs non-fraud
                fig = px.histogram(
                    data, 
                    x=feature_to_plot,
                    color='fraud',
                    barmode='overlay',
                    histnorm='percent',
                    labels={'fraud': 'Fraud'},
                    color_discrete_map={0: '#3498db', 1: '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.plotly_chart(create_correlation_heatmap(data), use_container_width=True)
                
                # Scatter plot for selected features
                st.subheader("Relationship Between Features")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("X-axis", data.columns.tolist(), index=0)
                
                with col2:
                    y_axis = st.selectbox("Y-axis", data.columns.tolist(), index=2)
                
                fig = px.scatter(
                    data, 
                    x=x_axis, 
                    y=y_axis,
                    color='fraud',
                    opacity=0.5,
                    color_discrete_map={0: '#3498db', 1: '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                feature_importance_fig = create_feature_importance_chart()
                if feature_importance_fig:
                    st.plotly_chart(feature_importance_fig, use_container_width=True)
                else:
                    st.warning("Feature importance chart not available. Make sure the model is properly loaded.")
            
            with tab4:
                st.subheader("Statistical Summary")
                st.dataframe(data.describe())
                
                # Class distribution
                fraud_stats = data['fraud'].value_counts(normalize=True) * 100
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Non-Fraud Transactions", f"{fraud_stats[0]:.2f}%")
                
                with col2:
                    st.metric("Fraud Transactions", f"{fraud_stats[1]:.2f}%")
        else:
            st.warning("Dataset not available for exploration.")
    
    # Model Performance page
    elif page == "Model Performance":
        st.title("Model Performance Analysis")
        
        st.markdown("""
        This page presents the performance metrics and visualizations for the four machine learning models
        that were evaluated for credit card fraud detection.
        """)
        
        # Tabs for the different models
        model_tabs = st.tabs([
            "Random Forest (Best)", 
            "Logistic Regression", 
            "Decision Tree", 
            "SGD Classifier"
        ])
        
        # Random Forest tab
        with model_tabs[0]:
            st.header("Random Forest Classifier")
            st.markdown("""
            The Random Forest model was selected as the best performing model for this project.
            It provided the highest accuracy and balanced precision/recall metrics.
            
            **Configuration:**
            - n_estimators: 50
            - max_depth: 5
            - min_samples_split: 50
            - min_samples_leaf: 20
            - max_features: 'sqrt'
            - class_weight: 'balanced'
            """)
            
            # Display images if available
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    st.image("confusion_matrix_Random Forest.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            
            with col2:
                try:
                    st.image("roc_curve_Random Forest.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance_fig = create_feature_importance_chart()
            if feature_importance_fig:
                st.plotly_chart(feature_importance_fig, use_container_width=True)
            
            # Metrics
            st.subheader("Performance Metrics")
            metrics_rf = {
                "Accuracy": 0.9998,
                "Precision": 0.9997,
                "Recall": 0.9996,
                "F1 Score": 0.9996,
                "AUC": 0.9998
            }
            
            # Create a DataFrame for metrics display
            metrics_df = pd.DataFrame({
                "Metric": list(metrics_rf.keys()),
                "Value": list(metrics_rf.values())
            })
            
            # Display metrics as a bar chart
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Value",
                color="Value",
                color_continuous_scale="Viridis",
                title="Key Performance Metrics",
                labels={"Value": "Score (0-1)"}
            )
            fig.update_layout(yaxis_range=[0.9, 1.0])
            st.plotly_chart(fig, use_container_width=True)
            
            # Data leakage explanation
            st.subheader("Data Leakage Analysis")
            st.markdown("""
            Despite excellent metrics, we discovered a significant data leakage issue:
            
            - The `ratio_to_median_purchase_price` feature had a 46% correlation with fraud
            - When we removed just this one feature, accuracy dropped from 99.98% to 48.16%
            - This indicated that a single feature was essentially "giving away" the answer
            
            This finding is crucial for real-world applications, as:
            1. Such a powerful single indicator might not be available in real-time
            2. Fraudsters could potentially learn to circumvent this single detection mechanism
            3. The model wasn't learning complex patterns but relying heavily on one feature
            """)
        
        # Logistic Regression tab
        with model_tabs[1]:
            st.header("Logistic Regression")
            st.markdown("""
            Logistic Regression provided a strong baseline model with good interpretability.
            
            **Configuration:**
            - C: 0.01 (strong regularization)
            - class_weight: 'balanced'
            - solver: 'liblinear'
            - max_iter: 1000
            """)
            
            # Display images if available
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    st.image("confusion_matrix_Logistic Regression.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            
            with col2:
                try:
                    st.image("roc_curve_Logistic Regression.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            
            # Metrics
            st.subheader("Performance Metrics")
            metrics_lr = {
                "Accuracy": 0.9994,
                "Precision": 0.9992,
                "Recall": 0.9990,
                "F1 Score": 0.9991,
                "AUC": 0.9995
            }
            
            # Create a DataFrame for metrics display
            metrics_df = pd.DataFrame({
                "Metric": list(metrics_lr.keys()),
                "Value": list(metrics_lr.values())
            })
            
            # Display metrics as a bar chart
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Value",
                color="Value",
                color_continuous_scale="Viridis",
                title="Key Performance Metrics",
                labels={"Value": "Score (0-1)"}
            )
            fig.update_layout(yaxis_range=[0.9, 1.0])
            st.plotly_chart(fig, use_container_width=True)
        
        # Decision Tree tab
        with model_tabs[2]:
            st.header("Decision Tree")
            st.markdown("""
            The Decision Tree model offered good explainability with competitive performance.
            
            **Configuration:**
            - max_depth: 4
            - min_samples_split: 50
            - min_samples_leaf: 20
            - class_weight: 'balanced'
            """)
            
            # Display images if available
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    st.image("confusion_matrix_Decision Tree.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            
            with col2:
                try:
                    st.image("roc_curve_Decision Tree.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            
            # Metrics
            st.subheader("Performance Metrics")
            metrics_dt = {
                "Accuracy": 0.9996,
                "Precision": 0.9994,
                "Recall": 0.9993,
                "F1 Score": 0.9993,
                "AUC": 0.9996
            }
            
            # Create a DataFrame for metrics display
            metrics_df = pd.DataFrame({
                "Metric": list(metrics_dt.keys()),
                "Value": list(metrics_dt.values())
            })
            
            # Display metrics as a bar chart
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Value",
                color="Value",
                color_continuous_scale="Viridis",
                title="Key Performance Metrics",
                labels={"Value": "Score (0-1)"}
            )
            fig.update_layout(yaxis_range=[0.9, 1.0])
            st.plotly_chart(fig, use_container_width=True)
        
        # SGD Classifier tab
        with model_tabs[3]:
            st.header("SGD Classifier (Linear SVM)")
            st.markdown("""
            The SGD Classifier (configured as a Linear SVM) offered fast training time with good scalability.
            
            **Configuration:**
            - loss: 'hinge' (SVM)
            - penalty: 'l2'
            - alpha: 0.01 (regularization strength)
            - class_weight: 'balanced'
            - max_iter: 1000
            """)
            
            # Display images if available
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    st.image("confusion_matrix_SGD Classifier.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            
            with col2:
                try:
                    st.image("roc_curve_SGD Classifier.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            
            # Metrics
            st.subheader("Performance Metrics")
            metrics_sgd = {
                "Accuracy": 0.9992,
                "Precision": 0.9990,
                "Recall": 0.9989,
                "F1 Score": 0.9989,
                "AUC": 0.9992
            }
            
            # Create a DataFrame for metrics display
            metrics_df = pd.DataFrame({
                "Metric": list(metrics_sgd.keys()),
                "Value": list(metrics_sgd.values())
            })
            
            # Display metrics as a bar chart
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Value",
                color="Value",
                color_continuous_scale="Viridis",
                title="Key Performance Metrics",
                labels={"Value": "Score (0-1)"}
            )
            fig.update_layout(yaxis_range=[0.9, 1.0])
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison section
        st.header("Model Comparison")
        
        # Create a DataFrame for comparison
        comparison_data = {
            "Model": ["Random Forest", "Logistic Regression", "Decision Tree", "SGD Classifier"],
            "Accuracy": [0.9998, 0.9994, 0.9996, 0.9992],
            "Precision": [0.9997, 0.9992, 0.9994, 0.9990],
            "Recall": [0.9996, 0.9990, 0.9993, 0.9989],
            "F1 Score": [0.9996, 0.9991, 0.9993, 0.9989],
            "AUC": [0.9998, 0.9995, 0.9996, 0.9992]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create a radar chart for model comparison
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        
        fig = go.Figure()
        
        for i, model in enumerate(comparison_df["Model"]):
            fig.add_trace(go.Scatterpolar(
                r=[comparison_df.iloc[i][cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.998, 1]
                )),
            title="Model Performance Comparison",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display comparison table
        st.subheader("Performance Metrics Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Display precision-recall curve if available
        st.subheader("Precision-Recall Curves")
        try:
            st.image("precision_recall_curve.png", caption="Precision-Recall Curves for All Models", use_container_width=True)
        except:
            st.warning("Precision-recall curve image not found.")
            
        # Final recommendation
        st.subheader("Model Selection Rationale")
        st.markdown("""
        **Random Forest was selected as the final model for the following reasons:**
        
        1. Highest overall performance across all metrics
        2. Better generalization to unseen data
        3. Provides feature importance for explainability
        4. More robust to the data leakage issue
        5. Good balance between precision and recall
        
        While all models showed exceptional performance due to the strong signal in the 
        `ratio_to_median_purchase_price` feature, Random Forest provided the most balanced approach
        and better handling of edge cases.
        """)
    
    # About page
    elif page == "About":
        st.title("About This Project")
        st.markdown("""
        ## Credit Card Fraud Detection Project
        
        This project implements machine learning models for credit card fraud detection, 
        with a focus on identifying and addressing data leakage issues while providing a practical application.
        
        ### Key Components:
        
        1. **Analysis Script** (`credit_card_fraud_detection.py`): Compares four models (Logistic Regression, Decision Tree, SGD, Random Forest)
        2. **Model Training** (`fraud_predictor.py`): Trains and saves the best model (Random Forest)
        3. **Prediction Interface** (`predict_fraud.py`): User-friendly interface for fraud prediction
        4. **Streamlit Dashboard** (`app.py`): Interactive visualization and prediction interface
        
        ### Key Findings: Data Leakage Issue
        
        During our analysis, we discovered an important issue that's common in machine learning:
        
        - All models achieved suspiciously high accuracy (99.98% for Random Forest)
        - The `ratio_to_median_purchase_price` feature had a 46% correlation with fraud
        - When we removed just this one feature, accuracy dropped dramatically from 99.98% to 48.16%
        
        ### Dataset Features:
        
        - `distance_from_home` - Distance from home where the transaction happened
        - `distance_from_last_transaction` - Distance from last transaction
        - `ratio_to_median_purchase_price` - Ratio of purchased price to median purchase price
        - `repeat_retailer` - Is the transaction from same retailer (1 for yes, 0 for no)
        - `used_chip` - Is the transaction through chip (1 for yes, 0 for no)
        - `used_pin_number` - Is the transaction using PIN number (1 for yes, 0 for no)
        - `online_order` - Is the transaction an online order (1 for yes, 0 for no)
        - `fraud` - Is the transaction fraudulent (target variable)
        """)
        
        # Display team or acknowledgments
        st.subheader("Acknowledgments")
        st.write("This project was created as a demonstration of machine learning applications in fraud detection.")

if __name__ == "__main__":
    main() 