"""
Main training pipeline orchestrator for Iris classification
Based on the reference MLOps architecture patterns
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import AmlCompute, Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, AzureCliCredential


def parse_args():
    parser = argparse.ArgumentParser("deploy-train")
    parser.add_argument("--model-name", type=str, default="iris-classifier", help="Model name")
    parser.add_argument("--experiment-name", type=str, default="iris-classification", help="Experiment name")
    parser.add_argument("--environment", type=str, default="Development", help="Environment (Development/Production)")
    parser.add_argument("--compute-name", type=str, default="ml-training-cluster", help="Compute cluster name")
    parser.add_argument("--wait-for-completion", action="store_true", help="Wait for job completion")
    
    return parser.parse_args()


def get_ml_client():
    """Get Azure ML Client"""
    try:
        # Try to load from config first
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        print("✅ Connected to Azure ML using config file")
    except Exception as e:
        print(f"⚠️  Config file not found: {e}")
        try:
            # Fallback to Azure CLI credential
            ml_client = MLClient.from_config(credential=AzureCliCredential())
            print("✅ Connected to Azure ML using Azure CLI")
        except Exception as e2:
            print(f"❌ Failed to connect to Azure ML: {e2}")
            sys.exit(1)
    
    return ml_client


def get_or_create_compute(ml_client, compute_name):
    """Get or create compute cluster"""
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"✅ Using existing compute: {compute_name}")
        return compute
    except Exception:
        print(f"🔄 Creating new compute cluster: {compute_name}")
        
        compute = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="Standard_DS3_v2",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=180,
            tier="Dedicated",
        )
        
        ml_client.compute.begin_create_or_update(compute)
        print(f"✅ Compute cluster {compute_name} created successfully")
        return compute


def get_or_create_environment(ml_client, environment_name="iris-training-env"):
    """Get or create custom environment"""
    try:
        # Try to get existing environment
        environment = ml_client.environments.get(environment_name, version="latest")
        print(f"✅ Using existing environment: {environment_name}")
        return environment
    except Exception:
        print(f"🔄 Creating new environment: {environment_name}")
        
        # Create environment from conda file
        conda_file_path = Path("environments") / "iris-training-env.yml"
        
        if not conda_file_path.exists():
            # Create a default conda file if it doesn't exist
            conda_content = """
name: iris-training-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - pip:
    - pandas>=2.0.0,<2.3.0
    - scikit-learn>=1.3.0
    - numpy>=1.24.0,<2.0.0
    - joblib>=1.3.0
    - azure-ai-ml>=1.30.0
    - azure-identity>=1.15.0
    - mlflow>=2.8.0
    - azureml-mlflow>=1.56.0
"""
            os.makedirs("environments", exist_ok=True)
            with open(conda_file_path, 'w') as f:
                f.write(conda_content)
            print(f"📝 Created conda environment file: {conda_file_path}")
        
        environment = Environment(
            name=environment_name,
            conda_file=str(conda_file_path),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            description="Iris classification training environment with MLflow"
        )
        
        created_env = ml_client.environments.create_or_update(environment)
        print(f"✅ Environment {environment_name} created successfully")
        return created_env


@pipeline(name="iris_training_pipeline", description="Complete Iris classification training pipeline")
def build_training_pipeline(model_name: str):
    """Build the complete training pipeline"""
    
    # Load components
    component_dir = Path("Pipeline/components")
    
    data_prep = load_component(source=str(component_dir / "data-preparation.yml"))
    feature_eng = load_component(source=str(component_dir / "feature-engineering.yml"))
    train_model = load_component(source=str(component_dir / "train-model.yml"))
    register_model = load_component(source=str(component_dir / "register-model.yml"))
    
    # Pipeline steps
    step_data_prep = data_prep(
        test_size=0.2,
        random_state=42
    )
    
    step_feature_eng = feature_eng(
        input_train_data=step_data_prep.outputs.output_train_data,
        input_test_data=step_data_prep.outputs.output_test_data
    )
    
    step_train = train_model(
        input_train_data=step_feature_eng.outputs.output_train_data,
        input_test_data=step_feature_eng.outputs.output_test_data,
        model_name=model_name
    )
    
    step_register = register_model(
        input_model_dir=step_train.outputs.output_model_dir,
        input_metrics_dir=step_train.outputs.output_metrics_dir,
        model_name=model_name,
        model_description=f"Iris classification model - {model_name}"
    )
    
    return {
        "trained_model": step_train.outputs.output_model_dir,
        "model_metrics": step_train.outputs.output_metrics_dir,
        "registration_status": step_register
    }


def submit_pipeline(ml_client, compute_name, environment, model_name, experiment_name):
    """Submit the training pipeline"""
    
    print(f"🚀 Building pipeline for model: {model_name}")
    
    # Create pipeline job
    pipeline_job = build_training_pipeline(model_name=model_name)
    
    # Configure pipeline settings
    pipeline_job.settings.default_compute = compute_name
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    pipeline_job.settings.force_rerun = True
    pipeline_job.display_name = f"Iris Training Pipeline - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📤 Submitting pipeline to experiment: {experiment_name}")
    
    # Submit job
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job, 
        experiment_name=experiment_name
    )
    
    print(f"✅ Job submitted successfully!")
    print(f"   - Job Name: {submitted_job.name}")
    print(f"   - Job ID: {submitted_job.id}")
    print(f"   - Experiment: {experiment_name}")
    print(f"   - Web View: https://ml.azure.com/runs/{submitted_job.name}")
    
    return submitted_job


def main():
    args = parse_args()
    
    print("🏗️  Starting Iris Classification Training Pipeline")
    print("=" * 60)
    print(f"Model Name: {args.model_name}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Environment: {args.environment}")
    print(f"Compute: {args.compute_name}")
    print("=" * 60)
    
    try:
        # Get Azure ML client
        ml_client = get_ml_client()
        
        # Get or create compute
        compute = get_or_create_compute(ml_client, args.compute_name)
        
        # Get or create environment
        environment = get_or_create_environment(ml_client)
        
        # Submit pipeline
        job = submit_pipeline(
            ml_client, 
            args.compute_name, 
            environment,
            args.model_name, 
            args.experiment_name
        )
        
        if args.wait_for_completion:
            print("⏳ Waiting for job completion...")
            ml_client.jobs.stream(job.name)
            
            # Check job status
            completed_job = ml_client.jobs.get(job.name)
            if completed_job.status == "Completed":
                print("✅ Pipeline completed successfully!")
            else:
                print(f"❌ Pipeline failed with status: {completed_job.status}")
                sys.exit(1)
        else:
            print("🔄 Pipeline submitted. Check Azure ML Studio for progress.")
        
        print("\n🎯 Pipeline Summary:")
        print(f"   - Job URL: https://ml.azure.com/runs/{job.name}")
        print(f"   - Experiment: {args.experiment_name}")
        print(f"   - Model: {args.model_name}")
        print("   - Components: Data Prep → Feature Engineering → Training → Registration")
        
    except Exception as e:
        print(f"❌ Pipeline submission failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()