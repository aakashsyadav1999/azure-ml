"""
Model registration component for Iris classification
Registers the trained model to Azure ML Model Registry
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser("register-model")
    parser.add_argument("--input_model_dir", type=str, help="Directory containing trained models")
    parser.add_argument("--input_metrics_dir", type=str, help="Directory containing model metrics")
    parser.add_argument("--model_name", type=str, default="iris-classifier", help="Model name for registration")
    parser.add_argument("--model_description", type=str, default="Iris classification model trained with multiple algorithms", help="Model description")
    
    return parser.parse_args()


def load_model_metadata(input_metrics_dir):
    """Load model metadata"""
    metadata_path = Path(input_metrics_dir) / "latest_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f" Loaded metadata: {metadata['model_name']}")
    print(f"   - Best model type: {metadata['best_model_type']}")
    print(f"   - Accuracy: {metadata['accuracy']:.4f}")
    print(f"   - Timestamp: {metadata['timestamp']}")
    
    return metadata


def register_with_mlflow(input_model_dir, model_name, metadata):
    """Register model using MLflow"""
    model_path = Path(input_model_dir) / "latest_model.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Start MLflow run for registration
    with mlflow.start_run(run_name=f"model_registration_{metadata['timestamp']}"):
        
        # Load and log the model
        import joblib
        model = joblib.load(model_path)
        
        # Log model parameters and metrics
        mlflow.log_param("model_type", metadata['best_model_type'])
        mlflow.log_param("timestamp", metadata['timestamp'])
        mlflow.log_metric("accuracy", metadata['accuracy'])
        
        # Log all model comparison results
        for model_type, results in metadata['all_results'].items():
            mlflow.log_metric(f"{model_type.lower()}_accuracy", results['accuracy'])
            mlflow.log_metric(f"{model_type.lower()}_f1_score", results['f1_score'])
        
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        print(f" Model registered with MLflow: {model_info.model_uri}")
        return model_info


def register_with_azure_ml(input_model_dir, model_name, model_description, metadata):
    """Register model with Azure ML Model Registry"""
    try:
        # Initialize Azure ML client
        credential = DefaultAzureCredential()
        
        # Try to get ML client from environment/config
        ml_client = MLClient.from_config(credential=credential)
        
        model_path = Path(input_model_dir) / "latest_model.joblib"
        
        # Create model entity
        model_entity = Model(
            name=model_name,
            path=str(model_path),
            description=f"{model_description}. Best model: {metadata['best_model_type']} (Accuracy: {metadata['accuracy']:.4f})",
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                "best_model_type": metadata['best_model_type'],
                "accuracy": str(metadata['accuracy']),
                "timestamp": metadata['timestamp'],
                "framework": "scikit-learn",
                "task": "classification",
                "dataset": "iris"
            }
        )
        
        # Register the model
        registered_model = ml_client.models.create_or_update(model_entity)
        
        print(f" Model registered with Azure ML:")
        print(f"   - Name: {registered_model.name}")
        print(f"   - Version: {registered_model.version}")
        print(f"   - ID: {registered_model.id}")
        
        return registered_model
        
    except Exception as e:
        print(f"  Azure ML registration failed: {str(e)}")
        print("   Model registration will continue with MLflow only")
        return None


def main():
    args = parse_args()
    
    print("Starting model registration...")
    print(f"Input model directory: {args.input_model_dir}")
    print(f"Input metrics directory: {args.input_metrics_dir}")
    print(f"Model name: {args.model_name}")
    
    # Load model metadata
    metadata = load_model_metadata(args.input_metrics_dir)
    
    # Register with MLflow (always works)
    mlflow_info = register_with_mlflow(args.input_model_dir, args.model_name, metadata)
    
    # Try to register with Azure ML (may fail in local environments)
    azure_ml_model = register_with_azure_ml(
        args.input_model_dir, 
        args.model_name, 
        args.model_description, 
        metadata
    )
    
    # Create registration summary
    registration_summary = {
        "model_name": args.model_name,
        "registration_timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "mlflow_registration": {
            "model_uri": str(mlflow_info.model_uri) if mlflow_info else None,
            "run_id": mlflow_info.run_id if mlflow_info else None
        },
        "azure_ml_registration": {
            "model_id": azure_ml_model.id if azure_ml_model else None,
            "model_version": azure_ml_model.version if azure_ml_model else None,
            "status": "success" if azure_ml_model else "failed"
        }
    }
    
    # Save registration summary
    summary_path = Path(args.input_metrics_dir) / "registration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(registration_summary, f, indent=2, default=str)
    
    print(f" Registration summary saved: {summary_path}")
    print(" Model registration completed successfully!")


if __name__ == "__main__":
    main()