"""
Create a small sample dataset for deployment demonstration
"""
import pandas as pd
import os

print("Creating sample dataset for deployment...")

try:
    # Try with 'copy' in the name first
    file_path = "card_transdata copy.csv"
    if not os.path.exists(file_path):
        file_path = "card_transdata.csv"
    
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Original dataset: {len(df)} rows, fraud cases: {df['fraud'].sum()} ({df['fraud'].mean()*100:.2f}%)")
    
    # Create a stratified sample to ensure we have some fraud cases
    # First, separate fraud and non-fraud
    fraud_df = df[df['fraud'] == 1]
    non_fraud_df = df[df['fraud'] == 0]
    
    # Sample from each
    fraud_sample = fraud_df.sample(n=min(100, len(fraud_df)), random_state=42)
    non_fraud_sample = non_fraud_df.sample(n=900, random_state=42)
    
    # Combine and shuffle
    sample_df = pd.concat([fraud_sample, non_fraud_sample]).sample(frac=1, random_state=42)
    
    # Save the sample
    sample_df.to_csv("card_transdata_sample.csv", index=False)
    print(f"Created sample dataset: {len(sample_df)} rows, fraud cases: {sample_df['fraud'].sum()} ({sample_df['fraud'].mean()*100:.2f}%)")
    print("Saved as card_transdata_sample.csv")
    
except Exception as e:
    print(f"Error creating sample dataset: {e}") 