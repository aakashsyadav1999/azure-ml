"""
Model training component for Iris classification
Trains multiple models and compares performance with MLflow tracking
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report)
from sklearn.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser("train-model")
    parser.add_argument("--input_train_data", type=str, help="Path to training data", default="./data")
    parser.add_argument("--input_test_data", type=str, help="Path to test data", default="./data")
    parser.add_argument("--output_model_dir", type=str, help="Directory to save trained models", default="./outputs")
    parser.add_argument("--output_metrics_dir", type=str, help="Directory to save metrics", default="./outputs")
    parser.add_argument("--model_name", type=str, default="iris-classifier", help="Model name for MLflow")
    
    return parser.parse_args()


def load_data(input_train_data, input_test_data):
    """Load training and test data"""
    
    # Check if input data exists, if not generate it
    if not os.path.exists(input_train_data) or not any(f.endswith('.csv') for f in os.listdir(input_train_data) if os.path.isfile(os.path.join(input_train_data, f))):
        print("📊 Input data not found, generating Iris dataset...")
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        
        # Create data directories
        os.makedirs(input_train_data, exist_ok=True)
        os.makedirs(input_test_data, exist_ok=True)
        
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create DataFrames
        feature_names = iris.feature_names
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df['target'] = y_train
        
        test_df = pd.DataFrame(X_test, columns=feature_names)
        test_df['target'] = y_test
        
        # Save to files
        train_df.to_csv(os.path.join(input_train_data, 'train_data.csv'), index=False)
        test_df.to_csv(os.path.join(input_test_data, 'test_data.csv'), index=False)
        
        print(f"✅ Generated training data: {train_df.shape}")
        print(f"✅ Generated test data: {test_df.shape}")
        
        return train_df, test_df
    
    # Load training data
    train_files = [f for f in os.listdir(input_train_data) if f.endswith('.csv')]
    train_dfs = []
    for file in train_files:
        df = pd.read_csv(Path(input_train_data) / file)
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # Load test data
    test_files = [f for f in os.listdir(input_test_data) if f.endswith('.csv')]
    test_dfs = []
    for file in test_files:
        df = pd.read_csv(Path(input_test_data) / file)
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Prepare features and targets"""
    # Separate features and targets
    feature_columns = [col for col in train_df.columns if col != 'species' and col != 'species_name']
    
    X_train = train_df[feature_columns]
    y_train = train_df['species']
    
    X_test = test_df[feature_columns]
    y_test = test_df['species']
    
    print(f"Feature columns: {feature_columns}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_columns


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'use_scaling': False
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'use_scaling': True
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'use_scaling': True
        }
    }
    
    results = {}
    trained_models = {}
    
    for model_name, config in models.items():
        print(f"\n🔄 Training {model_name}...")
        
        with mlflow.start_run(nested=True, run_name=f"iris_{model_name.lower()}"):
            # Log model type
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("uses_scaling", config['use_scaling'])
            
            # Create pipeline
            if config['use_scaling']:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', config['model'])
                ])
            else:
                pipeline = Pipeline([
                    ('classifier', config['model'])
                ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))
            
            # Log model parameters
            if hasattr(config['model'], 'get_params'):
                params = config['model'].get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(f"model_{param_name}", param_value)
            
            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': class_report
            }
            
            trained_models[model_name] = pipeline
            
            # Log model artifact
            mlflow.sklearn.log_model(pipeline, f"model_{model_name.lower()}")
            
            print(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results, trained_models


def save_models_and_metrics(trained_models, results, output_model_dir, output_metrics_dir, model_name):
    """Save the best model and all metrics"""
    
    # Find best model based on accuracy
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # Create output directories
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.joblib"
    latest_model_filename = "latest_model.joblib"
    
    model_path = Path(output_model_dir) / model_filename
    latest_model_path = Path(output_model_dir) / latest_model_filename
    
    joblib.dump(best_model, model_path)
    joblib.dump(best_model, latest_model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'best_model_type': best_model_name,
        'timestamp': timestamp,
        'accuracy': best_accuracy,
        'all_results': results,
        'model_filename': model_filename
    }
    
    metadata_filename = f"metadata_{timestamp}.json"
    latest_metadata_filename = "latest_metadata.json"
    
    metadata_path = Path(output_metrics_dir) / metadata_filename
    latest_metadata_path = Path(output_metrics_dir) / latest_metadata_filename
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    with open(latest_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Best model saved: {model_path}")
    print(f"✅ Latest model link: {latest_model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    print(f"✅ Latest metadata link: {latest_metadata_path}")
    
    return best_model, metadata


def main():
    args = parse_args()
    
    print("Starting model training...")
    print(f"Input training data: {args.input_train_data}")
    print(f"Input test data: {args.input_test_data}")
    print(f"Output model directory: {args.output_model_dir}")
    print(f"Output metrics directory: {args.output_metrics_dir}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"iris_classification_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Load data
        train_df, test_df = load_data(args.input_train_data, args.input_test_data)
        
        # Prepare features
        X_train, X_test, y_train, y_test, feature_columns = prepare_features(train_df, test_df)
        
        # Log dataset info
        mlflow.log_param("num_features", len(feature_columns))
        mlflow.log_param("num_classes", len(y_train.unique()))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Train models
        results, trained_models = train_models(X_train, X_test, y_train, y_test)
        
        # Save models and metrics
        best_model, metadata = save_models_and_metrics(
            trained_models, results, 
            args.output_model_dir, args.output_metrics_dir, 
            args.model_name
        )
        
        # Log best model metrics to main run
        best_model_name = metadata['best_model_type']
        best_results = results[best_model_name]
        
        mlflow.log_metric("best_accuracy", best_results['accuracy'])
        mlflow.log_metric("best_f1_score", best_results['f1_score'])
        mlflow.log_param("best_model_type", best_model_name)
        
        # Log final model
        mlflow.sklearn.log_model(best_model, "final_model")
        
    print("✅ Model training completed successfully!")


if __name__ == "__main__":
    main()