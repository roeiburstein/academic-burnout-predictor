import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    print("--- Phase 8 & 9: Model Training, Evaluation & Export ---")
    
    # 1. Load Cleaned Data
    csv_path = "cleaned_burnout_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run eda.py first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Target Encoding
    # We will predict binary "High Burnout" (1 = High, 0 = Medium or Low)
    df['target'] = (df['burnout_level'] == 'High').astype(int)
    print(f"Target distribution (High Burnout = 1, Other = 0):")
    print(df['target'].value_counts(normalize=True).round(4) * 100)
    
    # Drop columns that are not predictors or are the raw target
    df_features = df.drop(columns=['student_id', 'burnout_level', 'target'])
    
    # 3. Preprocessing Categorical Columns
    # Ordinal Mapping
    ordinal_mappings = {
        'year': {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4},
        'stress_level': {'Low': 1, 'Medium': 2, 'High': 3},
        'sleep_quality': {'Poor': 1, 'Average': 2, 'Good': 3},
        'internet_quality': {'Poor': 1, 'Average': 2, 'Good': 3}
    }
    
    for col, mapping in ordinal_mappings.items():
        df_features[col] = df_features[col].map(mapping)
        
    # Nominal Encoding (One-Hot)
    # We'll use gender (reference: Male) and course (reference: BCA)
    # To keep the web calculator clean and reproducible, we will record the dummies explicitly
    nominal_cols = ['gender', 'course']
    df_features = pd.get_dummies(df_features, columns=nominal_cols, drop_first=True)
    
    # Convert all columns to float for consistency
    for col in df_features.columns:
        df_features[col] = df_features[col].astype(float)
        
    feature_names = list(df_features.columns)
    print(f"\nModel features ({len(feature_names)}): {feature_names}")
    
    X = df_features.values
    y = df['target'].values
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store scaling parameters for export
    # scaler.mean_ and scaler.scale_ (which is standard deviation)
    scaling_params = {}
    for i, col_name in enumerate(feature_names):
        scaling_params[col_name] = {
            "mean": float(scaler.mean_[i]),
            "std": float(scaler.scale_[i])
        }
        
    # 6. Model Training & Evaluation
    
    # --- Model 1: Logistic Regression ---
    print("\n--- Training Model 1: Logistic Regression ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))
    print(classification_report(y_test, y_pred_lr))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    
    # --- Model 2: Decision Tree ---
    print("\n--- Training Model 2: Decision Tree ---")
    dec_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    dec_tree.fit(X_train_scaled, y_train)
    y_pred_dt = dec_tree.predict(X_test_scaled)
    print("Decision Tree Accuracy:", round(accuracy_score(y_test, y_pred_dt), 4))
    print(classification_report(y_test, y_pred_dt))
    
    # --- Model 3: Random Forest ---
    print("\n--- Training Model 3: Random Forest ---")
    rand_forest = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rand_forest.fit(X_train_scaled, y_train)
    y_pred_rf = rand_forest.predict(X_test_scaled)
    print("Random Forest Accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
    print(classification_report(y_test, y_pred_rf))
    
    # 7. Feature Importance (Logistic Regression Coefficients)
    coefficients = log_reg.coef_[0]
    intercept = float(log_reg.intercept_[0])
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Coefficient': np.abs(coefficients)
    }).sort_values(by='Absolute_Coefficient', ascending=False)
    
    print("\n--- Logistic Regression Feature Coefficients (Impact on Burnout) ---")
    print(importance_df.round(4).to_string(index=False))
    
    # 8. Export Model Assets to JSON for Web Calculator
    model_assets = {
        "model_name": "Logistic Regression (Binary High Burnout)",
        "intercept": intercept,
        "coefficients": {col_name: float(coefficients[i]) for i, col_name in enumerate(feature_names)},
        "scaling": scaling_params,
        "categorical_mappings": ordinal_mappings,
        "accuracy": float(accuracy_score(y_test, y_pred_lr))
    }
    
    with open("model_assets.json", "w") as f:
        json.dump(model_assets, f, indent=4)
        
    print("\nModel assets successfully saved to: model_assets.json")
    print("--- Model Training & Export Completed ---")

if __name__ == "__main__":
    main()
