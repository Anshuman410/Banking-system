import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score
import joblib
import os
import shap

def train():
    print("Loading data...")
    try:
        df = pd.read_csv('data/churn_data.csv')
    except FileNotFoundError:
        print("Dataset not found. Please run data_gen.py first.")
        return

    # Feature Engineering
    df['Balance_to_Salary_Ratio'] = df['Balance'] / df['EstimatedSalary']
    df['IsActive_by_CreditCard'] = df['IsActiveMember'] * df['HasCrCard']

    X = df.drop(['CustomerId', 'Exited'], axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'EstimatedSalary', 'Balance_to_Salary_Ratio', 'IsActive_by_CreditCard',
                        'MonthlyRevenue', 'NPS', 'CLV']
    categorical_features = ['Geography', 'Gender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])

    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    num_names = numeric_features
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = num_names + list(cat_names)
    
    # Convert to dense if sparse, easier for SHAP
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    print("Training XGBoost...")
    # Using scale_pos_weight for class imbalance
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    
    xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    # Simplified Grid Search for speed
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100]
    }
    
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='recall', n_jobs=-1)
    grid.fit(X_train_processed, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Params: {grid.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test_processed)
    y_proba = best_model.predict_proba(X_test_processed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Building K-Means Segmentation model on Churners...")
    # Train KMeans only on people likely to churn (for retention strategy mapping)
    # We'll use the processed training data of actual churners
    churner_idx = (y_train == 1)
    X_train_churners = X_train_processed[churner_idx]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_churners)

    print("Calculating SHAP values for Explainer...")
    # We use a background dataset for TreeExplainer
    background = shap.sample(X_train_processed, 100)
    explainer = shap.TreeExplainer(best_model, background)
    
    print("Saving models and artifacts...")
    os.makedirs('models', exist_ok=True)
    
    # Save the full pipeline for easy prediction
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    
    joblib.dump(full_pipeline, 'models/churn_model.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(best_model, 'models/xgboost_model.pkl')
    
    print("Training complete!")

if __name__ == "__main__":
    train()
