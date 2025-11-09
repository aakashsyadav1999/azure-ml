"""
ML model training script for Iris classification with Azure ML logging
"""
import pandas as pd
import numpy as np
import os
import argparse
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import mlflow

def load_data(data_path):
    """Load training data from CSV"""
    df = pd.read_csv(data_path)
    target_col = 'target'
    feature_cols = [col for col in df.columns if col not in [target_col, 'target_name']]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_cols

def train_model(X, y, model_type='random_forest', random_state=42):
    """Train classification model and log metrics to Azure ML"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=200, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
    # Log model hyperparameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("random_state", random_state)
    if model_type == 'random_forest':
        mlflow.log_param("n_estimators", 100)
    elif model_type == 'logistic_regression':
        mlflow.log_param("max_iter", 200)
    
    model.fit(X_train, y_train)
    
    # Get predictions and metrics
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
        'model_type': model_type,
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }
    
    # Log metrics to Azure ML
    mlflow.log_metric("train_accuracy", metrics['train_accuracy'])
    mlflow.log_metric("train_f1_macro", metrics['train_f1_macro'])
    mlflow.log_metric("val_accuracy", metrics['val_accuracy'])
    mlflow.log_metric("val_f1_macro", metrics['val_f1_macro'])
    mlflow.log_metric("train_samples", metrics['train_samples'])
    mlflow.log_metric("val_samples", metrics['val_samples'])
    
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Validation F1: {metrics['val_f1_macro']:.4f}")
    
    # Log classification report as artifact
    report = classification_report(y_val, y_val_pred)
    print(f"\nValidation classification report:\n{report}")
    
    # Save classification report as artifact
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("outputs/classification_report.txt")
    
    return model, metrics

def save_model(model, metrics, feature_cols, output_dir):
    """Save model and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"model_{timestamp}.joblib")
    metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_file': f"model_{timestamp}.joblib",
        'feature_columns': feature_cols,
        'metrics': metrics,
        'training_date': datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save latest versions
    latest_model_path = os.path.join(output_dir, "latest_model.joblib")
    latest_metadata_path = os.path.join(output_dir, "latest_metadata.json")
    
    joblib.dump(model, latest_model_path)
    with open(latest_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    
    return model_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data-path', required=True, help='Path to training CSV')
    parser.add_argument('--model-type', default='random_forest', 
                        choices=['random_forest', 'logistic_regression'])
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--random-state', type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"Training {args.model_type} model...")
    
    # Start MLflow tracking for Azure ML
    mlflow.start_run()
    
    try:
        # Load data
        X, y, feature_cols = load_data(args.data_path)
        
        # Log data info
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("features", len(feature_cols))
        mlflow.log_param("samples", len(X))
        mlflow.log_param("output_dir", args.output_dir)
        
        # Train model
        model, metrics = train_model(X, y, args.model_type, args.random_state)
        
        # Save model
        model_path, metadata_path = save_model(model, metrics, feature_cols, args.output_dir)
        
        # Log model as MLflow artifact
        mlflow.log_artifact(model_path.replace("outputs/", ""))
        mlflow.log_artifact(metadata_path.replace("outputs/", ""))
        
        print(f"Training complete! Validation accuracy: {metrics['val_accuracy']:.4f}")
        
    finally:
        # End MLflow tracking
        mlflow.end_run()

if __name__ == "__main__":
    main()
