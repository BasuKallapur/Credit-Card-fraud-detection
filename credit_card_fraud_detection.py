import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import parallel_backend  # Added for better performance
import warnings
import time
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
# Use more portable path resolution
try:
    # First try the local directory
    df = pd.read_csv("card_transdata.csv")
except FileNotFoundError:
    try:
        # Then try with "copy" in the name
        df = pd.read_csv("card_transdata copy.csv")
    except FileNotFoundError:
        # Fall back to the original path if needed
        print("Warning: Using hardcoded path as local file not found")
        df = pd.read_csv(r"C:\Users\basuk\Desktop\ML AAT\Credit-Card-fraud-detection\card_transdata copy.csv")

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFeature Information:")
print(df.info())
print("\nCheck for missing values:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df['fraud'].value_counts())
print(f"Percentage of fraud cases: {df['fraud'].mean() * 100:.2f}%")

# Exploratory Data Analysis (EDA)
print("\nGenerating correlation matrix...")
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.savefig('correlation_matrix.png')
plt.close()
print("Correlation matrix saved as 'correlation_matrix.png'")

# Check for potential data leakage by examining correlations with target
print("\nAnalyzing correlations with fraud target for potential data leakage...")
correlations = df.corr()['fraud'].sort_values(ascending=False)
print("Correlations with fraud target:")
print(correlations)

# If any feature has very high correlation (>0.5), it might indicate data leakage
high_corr_features = correlations[abs(correlations) > 0.5].index.tolist()
high_corr_features.remove('fraud')  # Remove the target itself
if high_corr_features:
    print(f"\nWARNING: Potential data leakage detected! These features have high correlation with fraud:")
    for feature in high_corr_features:
        print(f"- {feature}: {correlations[feature]:.4f}")
    print("Consider removing these features for a more realistic model.")
else:
    print("\nNo suspicious features with very high correlation to fraud detected.")

# Separate features and target
print("\nSplitting data into features and target...")
X = df.drop('fraud', axis=1)
y = df['fraud']

# Split the data into train, validation, and test sets
print("Splitting data into training, validation, and test sets...")
# First split: 80% train+validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Second split: 75% train, 25% validation (from the remaining 80%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# Feature scaling
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_val_scaled = scaler.transform(X_val)  # Apply same transformation to validation
X_test_scaled = scaler.transform(X_test)  # Apply same transformation to test
print("Feature scaling complete")

# Function to evaluate models
def evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test, model_name):
    print(f"Evaluating {model_name}...")
    
    # Evaluate on validation set first
    print(f"\nValidation set performance for {model_name}:")
    start_time = time.time()
    y_val_pred = model.predict(X_val_scaled)
    val_pred_time = time.time() - start_time
    print(f"Validation prediction completed in {val_pred_time:.2f} seconds")
    
    # Calculate validation accuracy
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")
    
    # Print validation classification report
    print(f"\n{model_name} Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    # Now evaluate on test set
    print(f"\nTest set performance for {model_name}:")
    start_time = time.time()
    y_test_pred = model.predict(X_test_scaled)
    test_pred_time = time.time() - start_time
    print(f"Test prediction completed in {test_pred_time:.2f} seconds")
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    
    # Print test classification report
    print(f"\n{model_name} Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Plot confusion matrix for test set
    print(f"Generating confusion matrix for {model_name} (test set)...")
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Replace spaces with underscores in the filename
    safe_model_name = model_name.replace(" ", "_")
    plt.savefig(f'confusion_matrix_{safe_model_name}.png')
    plt.close()
    print(f"Confusion matrix saved as 'confusion_matrix_{safe_model_name}.png'")
    
    # Calculate ROC curve and AUC if the model supports probability predictions
    try:
        print(f"Generating ROC curve for {model_name} (test set)...")
        y_score = model.decision_function(X_test_scaled) if hasattr(model, 'decision_function') else model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} (Test Set)')
        plt.legend(loc='lower right')
        # Replace spaces with underscores in the filename
        plt.savefig(f'roc_curve_{safe_model_name}.png')
        plt.close()
        print(f"ROC curve saved as 'roc_curve_{safe_model_name}.png'")
    except:
        print(f"ROC curve not available for {model_name}")
    
    # Calculate generalization error (difference between validation and test accuracy)
    gen_error = abs(val_accuracy - test_accuracy)
    print(f"Generalization error (|val_acc - test_acc|): {gen_error:.4f}")
    if gen_error > 0.05:
        print("WARNING: High generalization error. Model may be overfitting.")
    
    return test_accuracy, y_test_pred

print("\n" + "="*80)
print("TRAINING ALL MODELS WITH PARALLEL BACKEND")
print("="*80)

# Train all models with parallel backend for better performance
with parallel_backend('threading', n_jobs=-1):
    # Model 1: Logistic Regression
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*80)
    # Parameters for preventing overfitting
    print("Parameter grid for Logistic Regression (with strong regularization):")
    param_grid_lr = {
        'C': [0.01, 0.1],  # Smaller C values for stronger regularization
        'class_weight': ['balanced']
    }
    print(param_grid_lr)

    print("\nTraining Logistic Regression model (this may take a few minutes)...")
    start_time = time.time()
    lr_model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'), 
                           param_grid_lr, cv=3, scoring='f1', n_jobs=-1, verbose=2)  # Reduced CV folds
    lr_model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"\nLogistic Regression training completed in {train_time:.2f} seconds")
    print(f"Best parameters: {lr_model.best_params_}")
    lr_accuracy, lr_preds = evaluate_model(lr_model.best_estimator_, X_val_scaled, y_val, X_test_scaled, y_test, "Logistic Regression")

    # Model 2: Decision Tree Classifier
    print("\n" + "="*80)
    print("TRAINING DECISION TREE CLASSIFIER")
    print("="*80)
    # Parameters to strongly prevent overfitting
    print("Parameter grid for Decision Tree (with strict regularization to prevent overfitting):")
    param_grid_dt = {
        'max_depth': [3, 4],  # Very limited depth to prevent memorizing the data
        'min_samples_split': [50, 100],  # Require many samples per split
        'min_samples_leaf': [20, 50],  # Require large leaf nodes
        'class_weight': ['balanced']
    }
    print(param_grid_dt)

    print("\nTraining Decision Tree model (this may take a few minutes)...")
    start_time = time.time()
    dt_model = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                           param_grid_dt, cv=3, scoring='f1', n_jobs=-1, verbose=2)  # Reduced CV folds
    dt_model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"\nDecision Tree training completed in {train_time:.2f} seconds")
    print(f"Best parameters: {dt_model.best_params_}")
    dt_accuracy, dt_preds = evaluate_model(dt_model.best_estimator_, X_val_scaled, y_val, X_test_scaled, y_test, "Decision Tree")

    # Model 3: SGD Classifier (configured as Linear SVM)
    print("\n" + "="*80)
    print("TRAINING SGD CLASSIFIER (LINEAR SVM)")
    print("="*80)
    # Linear SVM-like SGD parameter grid
    print("Parameter grid for SGD Classifier (Linear SVM):")
    param_grid_sgd = {
        'alpha': [0.01, 0.1],  # Stronger regularization
        'class_weight': ['balanced'],
        'max_iter': [1000]
    }
    print(param_grid_sgd)

    print("\nTraining SGD Classifier model (this may take a few minutes)...")
    start_time = time.time()
    sgd_model = GridSearchCV(SGDClassifier(loss='hinge', penalty='l2', random_state=42), 
                            param_grid_sgd, cv=3, scoring='f1', n_jobs=-1, verbose=2)  # Reduced CV folds
    sgd_model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"\nSGD Classifier training completed in {train_time:.2f} seconds")
    print(f"Best parameters: {sgd_model.best_params_}")
    sgd_accuracy, sgd_preds = evaluate_model(sgd_model.best_estimator_, X_val_scaled, y_val, X_test_scaled, y_test, "SGD Classifier")

    # Model 4: Random Forest
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*80)
    # Parameters to strongly prevent overfitting
    print("Parameter grid for Random Forest (with strict regularization to prevent overfitting):")
    param_grid_rf = {
        'n_estimators': [50],  # Fewer trees for faster execution
        'max_depth': [3, 5],  # Very limited depth
        'min_samples_split': [50],  # Require many samples per split
        'min_samples_leaf': [20],  # Require large leaf nodes
        'max_features': ['sqrt'],  # Limit features considered at each split
        'class_weight': ['balanced']
    }
    print(param_grid_rf)

    print("\nTraining Random Forest model (this may take a few minutes)...")
    start_time = time.time()
    rf_model = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=2)  # Reduced CV folds
    rf_model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"\nRandom Forest training completed in {train_time:.2f} seconds")
    print(f"Best parameters: {rf_model.best_params_}")
    rf_accuracy, rf_preds = evaluate_model(rf_model.best_estimator_, X_val_scaled, y_val, X_test_scaled, y_test, "Random Forest")

# Feature importance for Random Forest
print("\nGenerating feature importance plot for Random Forest...")
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Feature importance plot saved as 'feature_importance.png'")

# Compare model performance
print("\n" + "="*80)
print("COMPARING MODEL PERFORMANCE")
print("="*80)
models = ['Logistic Regression', 'Decision Tree', 'SGD Classifier', 'Random Forest']
accuracies = [lr_accuracy, dt_accuracy, sgd_accuracy, rf_accuracy]

print("\nGenerating model comparison plot...")
plt.figure(figsize=(12, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
plt.savefig('model_comparison.png')
plt.close()
print("Model comparison plot saved as 'model_comparison.png'")

# Print comparative analysis
print("\n--- Comparative Analysis ---")
model_results = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies
}).sort_values('Accuracy', ascending=False)
print(model_results)

# --- NEW: Save all metrics for each model to CSV ---
# Calculate all metrics for each model on the test set
metrics_data = []
model_objs = [lr_model.best_estimator_, dt_model.best_estimator_, sgd_model.best_estimator_, rf_model.best_estimator_]
model_names = ['Logistic Regression', 'Decision Tree', 'SGD Classifier', 'Random Forest']

for model, name in zip(model_objs, model_names):
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_score = model.decision_function(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    metrics_data.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC': auc
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('model_metrics.csv', index=False)
print("\nAll model metrics saved to 'model_metrics.csv'.")

# Save the best model
best_model_index = np.argmax(accuracies)
best_model_name = models[best_model_index]
print(f"\nBest Model: {best_model_name} with accuracy: {accuracies[best_model_index]:.4f}")

# Additional Analysis: Precision-Recall tradeoff for imbalanced data
from sklearn.metrics import precision_recall_curve, average_precision_score

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_test, y_scores, model_names):
    plt.figure(figsize=(10, 8))
    
    for i, (score, name) in enumerate(zip(y_scores, model_names)):
        try:
            precision, recall, _ = precision_recall_curve(y_test, score)
            avg_precision = average_precision_score(y_test, score)
            plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
        except:
            print(f"PR curve not available for {name}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Different Models')
    plt.legend(loc='best')
    plt.savefig('precision_recall_curve.png')
    plt.close()

print("\nGenerating Precision-Recall curves...")
# Get probability scores for models
try:
    lr_scores = lr_model.best_estimator_.predict_proba(X_test_scaled)[:, 1]
    dt_scores = dt_model.best_estimator_.predict_proba(X_test_scaled)[:, 1]
    # SGD with hinge loss doesn't have predict_proba, use decision_function instead
    sgd_scores = sgd_model.best_estimator_.decision_function(X_test_scaled)
    rf_scores = rf_model.best_estimator_.predict_proba(X_test_scaled)[:, 1]
    
    model_scores = [lr_scores, dt_scores, sgd_scores, rf_scores]
    plot_precision_recall_curve(y_test, model_scores, models)
    print("Precision-Recall curve saved as 'precision_recall_curve.png'")
except Exception as e:
    print(f"Error generating precision-recall curve: {e}")

# Additional Analysis: Try removing the most predictive feature to test for data leakage
print("\n" + "="*80)
print("TESTING MODEL WITHOUT RATIO_TO_MEDIAN_PURCHASE_PRICE FEATURE")
print("="*80)
print("Removing 'ratio_to_median_purchase_price' which has highest correlation with fraud (0.46)")

# Prepare data without the most predictive feature
X_reduced = X.drop('ratio_to_median_purchase_price', axis=1)
print(f"Reduced feature set: {list(X_reduced.columns)}")

# Split the data into train, validation, and test sets
X_temp_reduced, X_test_reduced, y_temp, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
X_train_reduced, X_val_reduced, y_train, y_val = train_test_split(X_temp_reduced, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Scale features
scaler_reduced = StandardScaler()
X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
X_val_reduced_scaled = scaler_reduced.transform(X_val_reduced)
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)

# Train a simplified Random Forest with reduced features
print("\nTraining Random Forest with reduced feature set...")
rf_reduced = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rf_reduced.fit(X_train_reduced_scaled, y_train)

# Evaluate on validation set
y_val_pred = rf_reduced.predict(X_val_reduced_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy without ratio_to_median_purchase_price: {val_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = rf_reduced.predict(X_test_reduced_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nTest Accuracy without ratio_to_median_purchase_price: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print(f"\nAccuracy difference: Original RF {rf_accuracy:.4f} vs. Reduced Features {test_accuracy:.4f}")
print(f"Accuracy drop: {rf_accuracy - test_accuracy:.4f}")

print("\nCONCLUSION:")
if rf_accuracy - test_accuracy > 0.05:
    print("The ratio_to_median_purchase_price feature significantly affects model performance.")
    print("This suggests this feature may contain information that makes fraud detection unrealistically easy.")
    print("For a more realistic model, consider creating a model without this feature.")
else:
    print("Removing the ratio_to_median_purchase_price feature does not significantly impact performance.")
    print("The model may be learning from other patterns in the data.")

print("\nAnalysis completed. All visualization files have been saved to the current directory.") 