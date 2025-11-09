"""
Model deployment script for Azure ML using SDK v2
Deploys trained model as a web service endpoint
"""

import os
import argparse
import yaml
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, ManagedOnlineDeployment, Model, Environment, CodeConfiguration
)
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


def get_latest_model(ml_client, model_name="iris-classifier"):
    """Get the latest version of the model"""
    try:
        models = ml_client.models.list(name=model_name)
        latest_model = max(models, key=lambda x: int(x.version))
        print(f"Found model: {latest_model.name}, version: {latest_model.version}")
        return latest_model
    except Exception as e:
        print(f"Model not found: {e}")
        return None


def create_scoring_script():
    """Create scoring script for inference"""
    scoring_script_content = '''
import json
import joblib
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "latest_model.joblib")
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = np.array(data)
        result = model.predict(data)
        probabilities = model.predict_proba(data)
        
        # Convert to regular Python types for JSON serialization
        predictions = result.tolist()
        probs = probabilities.tolist()
        
        return json.dumps({
            "predictions": predictions,
            "probabilities": probs,
            "class_names": ["setosa", "versicolor", "virginica"]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
'''
    
    # Save scoring script
    os.makedirs("deployment", exist_ok=True)
    scoring_script_path = "deployment/score.py"
    
    with open(scoring_script_path, 'w') as f:
        f.write(scoring_script_content)
    
    return scoring_script_path


def create_deployment_environment(ml_client):
    """Create environment for deployment"""
    conda_env_content = '''
name: iris-inference
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pip
  - pip:
    - scikit-learn>=1.3.0
    - numpy>=1.24.0,<2.0.0
    - joblib>=1.3.0
    - azure-ai-ml
'''
    
    # Create deployment directory if it doesn't exist
    deployment_dir = "deployment"
    os.makedirs(deployment_dir, exist_ok=True)
    
    conda_env_path = os.path.join(deployment_dir, "conda-env.yml")
    with open(conda_env_path, 'w') as f:
        f.write(conda_env_content)
    
    # Create environment
    environment = Environment(
        name="iris-inference-env",
        conda_file=conda_env_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        description="Iris classification inference environment"
    )
    
    try:
        env = ml_client.environments.create_or_update(environment)
        print(f"Environment created: {env.name}")
        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        # Try to get existing environment
        try:
            env = ml_client.environments.get("iris-inference-env", version="1")
            print(f"Using existing environment: {env.name}")
            return env
        except:
            raise e


def deploy_model(ml_client, model, environment, endpoint_name="iris-classifier-endpoint", deployment_name="iris-deployment"):
    """Deploy model as managed online endpoint"""
    
    # Create scoring script
    scoring_script_path = create_scoring_script()
    
    # Check if endpoint exists
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        print(f"Using existing endpoint: {endpoint_name}")
    except:
        print(f"Creating new endpoint: {endpoint_name}")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Iris species classification endpoint",
            tags={"model": "iris-classifier", "method": "sklearn"}
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        endpoint = ml_client.online_endpoints.get(endpoint_name)
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code="deployment",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
        description="Iris classification deployment"
    )
    
    print(f"Creating deployment: {deployment_name}")
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    # Set traffic to 100%
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    return endpoint, deployment


def test_endpoint(ml_client, endpoint_name):
    """Test the deployed endpoint"""
    # Sample Iris data for testing
    test_data = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Should be setosa
            [6.0, 3.0, 4.8, 1.8],  # Should be virginica
            [5.8, 2.7, 4.1, 1.0]   # Should be versicolor
        ]
    }
    
    print("Testing endpoint with sample data...")
    try:
        result = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=None,
            request_json=test_data
        )
        print(f"Test result: {result}")
        return result
    except Exception as e:
        print(f"Test failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Deploy Iris model to Azure ML')
    parser.add_argument('--config', default='config/azure-ml-config.yml')
    parser.add_argument('--model-name', default='iris-classifier')
    parser.add_argument('--endpoint-name', default='iris-classifier-endpoint')
    parser.add_argument('--deployment-name', default='iris-deployment')
    parser.add_argument('--tenant-id', help='Azure tenant ID')
    parser.add_argument('--client-id', help='Service principal client ID')
    parser.add_argument('--client-secret', help='Service principal client secret')
    parser.add_argument('--test-endpoint', action='store_true', help='Test the endpoint after deployment')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Authenticate
        credential = authenticate_azure_ml(
            args.tenant_id,
            args.client_id,
            args.client_secret
        )
        
        # Get ML client
        ml_client = get_ml_client(config, credential)
        
        # Get latest model
        model = get_latest_model(ml_client, args.model_name)
        if model is None:
            print("No trained model found. Please train a model first.")
            return
        
        # Create environment
        environment = create_deployment_environment(ml_client)
        
        # Deploy model
        endpoint, deployment = deploy_model(
            ml_client, model, environment, args.endpoint_name, args.deployment_name
        )
        
        print(f"Model deployed successfully!")
        print(f"Endpoint name: {endpoint.name}")
        print(f"Scoring URI: {endpoint.scoring_uri}")
        
        if args.test_endpoint:
            test_endpoint(ml_client, args.endpoint_name)
        
        # Save deployment info
        deployment_info = {
            "endpoint_name": endpoint.name,
            "scoring_uri": endpoint.scoring_uri,
            "deployment_name": args.deployment_name,
            "model_name": model.name,
            "model_version": model.version
        }
        
        os.makedirs("deployment", exist_ok=True)
        with open("deployment/deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"Deployment info saved to: deployment/deployment_info.json")
        
    except Exception as e:
        print(f"Error in deployment: {str(e)}")
        raise


if __name__ == "__main__":
    main()