"""
Azure ML pipeline script - submits training job to Azure ML using SDK v2
"""
import os
import argparse
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, ClientSecretCredential


def authenticate_azure_ml(tenant_id=None, client_id=None, client_secret=None):
    """Authenticate with Azure ML using SDK v2"""
    if all([tenant_id, client_id, client_secret]):
        print("Authenticating with Service Principal...")
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        return credential
    else:
        print("Using default authentication...")
        return DefaultAzureCredential()


def load_config(config_path):
    """Load Azure ML configuration"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_ml_client(config, credential):
    """Get Azure ML Client"""
    ws_config = config['workspace']
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=ws_config['subscription_id'],
        resource_group_name=ws_config['resource_group'],
        workspace_name=ws_config['name']
    )
    
    print(f"Connected to workspace: {ws_config['name']}")
    return ml_client


def get_or_create_compute(ml_client, config):
    """Get or create compute cluster using SDK v2"""
    compute_config = config['compute']
    compute_name = compute_config['name']
    
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"Compute cluster '{compute_name}' found.")
        return compute
    except Exception:
        print(f"Creating new compute cluster: {compute_name}")
        
        from azure.ai.ml.entities import AmlCompute
        
        compute = AmlCompute(
            name=compute_name,
            size=compute_config['vm_size'],
            min_instances=compute_config.get('min_instances', 0),
            max_instances=compute_config.get('max_instances', 1)
        )
        
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"Compute cluster '{compute_name}' created successfully.")
        return compute


def get_environment(ml_client, config):
    """Get or create environment using SDK v2"""
    env_config = config['environment']
    env_name = env_config['name']
    
    try:
        environment = ml_client.environments.get(env_name, version="1")
        print(f"Environment '{env_name}' found.")
        return environment
    except Exception:
        print(f"Creating new environment: {env_name}")
        
        from azure.ai.ml.entities import Environment
        
        environment = Environment(
            name=env_name,
            conda_file=env_config['conda_file'],
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            description="Iris classification training environment"
        )
        
        ml_client.environments.create_or_update(environment)
        print(f"Environment '{env_name}' created successfully.")
        return environment


def submit_training_job(ml_client, compute, environment, config):
    """Submit training job using SDK v2"""
    exp_config = config['experiment']
    
    from azure.ai.ml import command
    
    job = command(
        code="./",  # Upload the entire project
        command="python src/train_model.py --data-path data/train.csv --output-dir outputs",
        environment="iris-training-env:1",  # Environment name:version
        compute="ml-training-cluster",  # Compute name
        display_name="Iris Classification Training",
        experiment_name=exp_config['name'],
        description="Train Iris classification model with scikit-learn"
    )
    
    print("Submitting training job...")
    submitted_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted: {submitted_job.name}")
    
    return submitted_job


def register_model_after_training(ml_client, job, model_name="iris-classifier"):
    """Register the trained model after successful training"""
    if job.status == 'Completed':
        try:
            from azure.ai.ml.entities import Model
            from azure.ai.ml.constants import AssetTypes
            
            # Try multiple possible paths for the model
            model_paths = [
                f"azureml://jobs/{job.name}/outputs/artifacts/outputs/latest_model.joblib",
                f"azureml://jobs/{job.name}/outputs/default/outputs/latest_model.joblib",
                f"azureml://jobs/{job.name}/outputs/outputs/latest_model.joblib"
            ]
            
            for model_path in model_paths:
                try:
                    print(f"Trying to register model from path: {model_path}")
                    model = Model(
                        name=model_name,
                        path=model_path,
                        description="Iris classification model trained with scikit-learn",
                        type=AssetTypes.CUSTOM_MODEL,
                        tags={
                            "algorithm": "RandomForest",
                            "framework": "scikit-learn", 
                            "dataset": "iris"
                        }
                    )
                    
                    registered_model = ml_client.models.create_or_update(model)
                    print(f"Model registered: {registered_model.name}, version: {registered_model.version}")
                    return registered_model
                except Exception as path_error:
                    print(f"Failed with path {model_path}: {str(path_error)}")
                    continue
            
            print("Failed to register model with any of the attempted paths")
            return None
            
        except Exception as e:
            print(f"Failed to register model: {str(e)}")
            return None
    else:
        print(f"Training job status: {job.status}. Model not registered.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/azure-ml-config.yml')
    parser.add_argument('--wait-for-completion', action='store_true')
    parser.add_argument('--tenant-id', help='Azure tenant ID')
    parser.add_argument('--client-id', help='Service principal client ID')
    parser.add_argument('--client-secret', help='Service principal client secret')
    parser.add_argument('--register-model', action='store_true', help='Register model after training')
    
    args = parser.parse_args()
    
    # Use environment variables as fallback if not provided as arguments
    tenant_id = args.tenant_id or os.getenv('AZURE_TENANT_ID')
    client_id = args.client_id or os.getenv('AZURE_CLIENT_ID') 
    client_secret = args.client_secret or os.getenv('AZURE_CLIENT_SECRET')
    
    if not tenant_id or not client_id or not client_secret:
        print("Error: Azure credentials not provided. Use command line arguments or environment variables:")
        print("  --tenant-id or AZURE_TENANT_ID")
        print("  --client-id or AZURE_CLIENT_ID") 
        print("  --client-secret or AZURE_CLIENT_SECRET")
        return
    
    # Load config
    config = load_config(args.config)
    
    # Authenticate
    credential = authenticate_azure_ml(tenant_id, client_id, client_secret)
    
    # Get ML client
    ml_client = get_ml_client(config, credential)
    
    # Get compute
    compute = get_or_create_compute(ml_client, config)
    
    # Get environment
    environment = get_environment(ml_client, config)
    
    # Submit job
    job = submit_training_job(ml_client, compute, environment, config)
    
    if args.wait_for_completion:
        print("Waiting for completion...")
        ml_client.jobs.stream(job.name)
        
        # Get final job status
        completed_job = ml_client.jobs.get(job.name)
        print(f"Training completed with status: {completed_job.status}")
        
        # Register model if requested
        if args.register_model and completed_job.status == 'Completed':
            register_model_after_training(ml_client, completed_job)
    else:
        print("Job submitted! Check Azure ML Studio for progress.")


if __name__ == "__main__":
    main()