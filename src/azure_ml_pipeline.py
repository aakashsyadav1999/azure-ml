"""
Azure ML pipeline script - submits training job to Azure ML
"""
import os
import argparse
import yaml
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import ServicePrincipalAuthentication

def authenticate_azure_ml(tenant_id=None, service_principal_id=None, service_principal_password=None):
    """Authenticate with Azure ML"""
    if all([tenant_id, service_principal_id, service_principal_password]):
        print("Authenticating with Service Principal...")
        auth = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password
        )
        return auth
    else:
        print("Using default authentication...")
        return None

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/azure-ml-config.yml')
    parser.add_argument('--wait-for-completion', action='store_true')
    parser.add_argument('--tenant-id', help='Azure tenant ID')
    parser.add_argument('--service-principal-id', help='Service principal ID')
    parser.add_argument('--service-principal-password', help='Service principal password')
    
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
    else:
        print("Job submitted! Check Azure ML Studio for progress.")

if __name__ == "__main__":
    main()