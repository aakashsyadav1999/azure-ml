"""
Azure ML pipeline script - submits training job to Azure ML using SDK v2
"""
import os
import argparse
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job, Command, Environment, Model
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
            min_instances=0,
            max_instances=compute_config['max_nodes']
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
        code="./",  # Source directory
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
            
            # Register model from the job outputs
            model = Model(
                name=model_name,
                path=f"azureml://jobs/{job.name}/outputs/artifacts/paths/latest_model.joblib",
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
    
    # Load config
    config = load_config(args.config)
    
    # Authenticate
    credential = authenticate_azure_ml(
        args.tenant_id,
        args.client_id, 
        args.client_secret
    )
    
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

def load_config(config_path):
    """Load Azure ML configuration"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_workspace(config, auth=None):
    """Connect to Azure ML workspace"""
    ws_config = config['workspace']
    
    ws = Workspace.get(
        name=ws_config['name'],
        subscription_id=ws_config['subscription_id'],
        resource_group=ws_config['resource_group'],
        auth=auth
    )
    
    print(f"Connected to workspace: {ws.name}")
    return ws

def get_or_create_compute(ws, config):
    """Get or create compute target"""
    compute_config = config['compute']
    compute_name = compute_config['name']
    
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print(f"Using existing compute: {compute_name}")
    except ComputeTargetException:
        print(f"Creating compute: {compute_name}")
        
        compute_config_obj = AmlCompute.provisioning_configuration(
            vm_size=compute_config['vm_size'],
            max_nodes=compute_config['max_instances'],
            min_nodes=compute_config['min_instances']
        )
        
        compute_target = ComputeTarget.create(ws, compute_name, compute_config_obj)
        compute_target.wait_for_completion(show_output=True)
    
    return compute_target

def get_environment(ws, config):
    """Get or create environment"""
    env_config = config['environment']
    env_name = env_config['name']
    
    try:
        env = Environment.get(workspace=ws, name=env_name)
        print(f"Using existing environment: {env_name}")
    except:
        print(f"Creating environment: {env_name}")
        env = Environment.from_conda_specification(
            name=env_name,
            file_path=env_config['conda_file']
        )
        env.docker.enabled = True
        env.register(workspace=ws)
    
    return env

def submit_training_job(ws, compute_target, environment, config):
    """Submit training job"""
    exp_config = config['experiment']
    experiment = Experiment(workspace=ws, name=exp_config['name'])
    
    script_config = ScriptRunConfig(
        source_directory='.',  # Upload entire project directory
        script='src/train_model.py',
        arguments=[
            '--data-path', './data/train.csv',  # Relative to project root
            '--model-type', 'random_forest',
            '--output-dir', './models'
        ],
        compute_target=compute_target,
        environment=environment
    )
    
    run = experiment.submit(config=script_config)
    print(f"Training job submitted: {run.id}")
    print(f"Monitor at: {run.get_portal_url()}")
    
    return run

def register_model_after_training(ws, run, model_name="iris-classifier"):
    """Register the trained model after successful training"""
    if run.get_status() == 'Completed':
        try:
            # Register model from the run
            model = run.register_model(
                model_name=model_name,
                model_path="outputs/latest_model.joblib",
                description="Iris classification model trained with scikit-learn",
                tags={
                    "algorithm": "RandomForest",
                    "framework": "scikit-learn", 
                    "dataset": "iris",
                    "accuracy": str(run.get_metrics().get("validation_accuracy", "unknown"))
                },
                model_framework="ScikitLearn"
            )
            print(f"Model registered: {model.name}, version: {model.version}")
            return model
        except Exception as e:
            print(f"Failed to register model: {str(e)}")
            return None
    else:
        print(f"Training run status: {run.get_status()}. Model not registered.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/azure-ml-config.yml')
    parser.add_argument('--wait-for-completion', action='store_true')
    parser.add_argument('--tenant-id', help='Azure tenant ID')
    parser.add_argument('--service-principal-id', help='Service principal ID')
    parser.add_argument('--service-principal-password', help='Service principal password')
    parser.add_argument('--register-model', action='store_true', help='Register model after training')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Authenticate
    auth = authenticate_azure_ml(
        args.tenant_id,
        args.service_principal_id, 
        args.service_principal_password
    )
    
    # Get workspace
    ws = get_workspace(config, auth)
    
    # Get compute
    compute_target = get_or_create_compute(ws, config)
    
    # Get environment
    environment = get_environment(ws, config)
    
    # Submit job
    run = submit_training_job(ws, compute_target, environment, config)
    
    if args.wait_for_completion:
        print("Waiting for completion...")
        run.wait_for_completion(show_output=True)
        print(f"Training completed with status: {run.get_status()}")
        
        # Register model if requested
        if args.register_model:
            register_model_after_training(ws, run)
    else:
        print("Job submitted! Check Azure ML Studio for progress.")

if __name__ == "__main__":
    main()