"""
Credit Card Fraud Detection - Interactive Prediction Interface

This script provides a user-friendly command-line interface for predicting 
credit card fraud based on transaction details. It uses the trained model from
fraud_predictor.py to make predictions.

Features used for prediction:
- distance_from_home: Distance from home where transaction occurred (miles)
- distance_from_last_transaction: Distance from last transaction (miles)
- ratio_to_median_purchase_price: Ratio of purchase price to median
- repeat_retailer: Whether transaction occurred at a repeat retailer (1=yes, 0=no)
- used_chip: Whether chip was used (1=yes, 0=no)
- used_pin_number: Whether PIN was used (1=yes, 0=no)
- online_order: Whether it was an online order (1=yes, 0=no)

The script will:
1. Prompt the user to enter values for each feature
2. Make a prediction using the trained model
3. Display the result with fraud probability
4. Provide insights about risk factors
5. Allow testing multiple transactions

Usage:
    python predict_fraud.py
"""

from fraud_predictor import predict_transaction

def main():
    print("\n===== CREDIT CARD FRAUD DETECTION =====")
    print("Enter the details of the transaction:\n")
    
    try:
        # Get user inputs
        distance_home = float(input("Distance from home (miles): "))
        distance_last = float(input("Distance from last transaction (miles): "))
        ratio_median = float(input("Ratio to median purchase price (0.1-10): "))
        repeat_retailer = int(input("Repeat retailer (1 for yes, 0 for no): "))
        used_chip = int(input("Used chip (1 for yes, 0 for no): "))
        used_pin = int(input("Used PIN number (1 for yes, 0 for no): "))
        online_order = int(input("Online order (1 for yes, 0 for no): "))
        
        # Make prediction
        prediction, probability = predict_transaction(
            distance_from_home=distance_home,
            distance_from_last_transaction=distance_last,
            ratio_to_median_purchase_price=ratio_median,
            repeat_retailer=repeat_retailer,
            used_chip=used_chip,
            used_pin_number=used_pin,
            online_order=online_order
        )
        
        # Display result
        print("\n" + "="*50)
        if prediction == 1:
            print(f"⚠️ FRAUD ALERT: This transaction is predicted as FRAUDULENT")
            print(f"Fraud probability: {probability:.4f} ({probability*100:.2f}%)")
        else:
            print(f"✅ This transaction is predicted as LEGITIMATE")
            print(f"Fraud probability: {probability:.4f} ({probability*100:.2f}%)")
        print("="*50)
        
        # Provide insights
        print("\nINSIGHTS:")
        factors = []
        
        # Key risk factors
        if ratio_median > 2:
            factors.append(f"- Purchase amount is {ratio_median:.1f}x the median (high risk)")
        
        if online_order == 1:
            factors.append("- Online order (increased risk)")
            
        if distance_home > 100:
            factors.append(f"- Transaction occurred {distance_home:.1f} miles from home")
            
        if used_chip == 0 and used_pin == 0:
            factors.append("- No chip or PIN security features used")
            
        if not factors:
            print("- No significant risk factors identified")
        else:
            for factor in factors:
                print(factor)
    
    except ValueError:
        print("Error: Please enter valid numeric values")

if __name__ == "__main__":
    main()
    
    # Ask if user wants to try another transaction
    while True:
        again = input("\nWould you like to check another transaction? (y/n): ")
        if again.lower() != 'y':
            print("Thank you for using the Fraud Detection tool!")
            break
        main() 