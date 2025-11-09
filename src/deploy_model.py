"""
Model deployment script for Azure ML
Deploys trained model as a web service endpoint
"""

import os
import argparse
import yaml
import joblib
import json
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
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


def get_latest_model(ws, model_name="iris-classifier"):
    """Get the latest version of the model"""
    try:
        model = Model(ws, model_name)
        print(f"Found model: {model.name}, version: {model.version}")
        return model
    except Exception as e:
        print(f"Model not found: {e}")
        return None


def create_inference_config():
    """Create inference configuration"""
    # Create scoring script content
    scoring_script_content = '''
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("iris-classifier")
    model = joblib.load(model_path)

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
    
    # Create environment file
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
    - azureml-defaults
'''
    
    conda_env_path = "deployment/conda-env.yml"
    with open(conda_env_path, 'w') as f:
        f.write(conda_env_content)
    
    # Create inference config
    inference_config = InferenceConfig(
        entry_script=scoring_script_path,
        environment=Environment.from_conda_specification(
            name="iris-inference-env",
            file_path=conda_env_path
        )
    )
    
    return inference_config


def deploy_model(ws, model, inference_config, service_name="iris-classifier-service"):
    """Deploy model as web service"""
    
    # Check if service already exists
    try:
        existing_service = ws.webservices[service_name]
        print(f"Updating existing service: {service_name}")
        existing_service.update(model=[model], inference_config=inference_config)
        existing_service.wait_for_deployment(show_output=True)
        return existing_service
    except KeyError:
        print(f"Creating new service: {service_name}")
        pass
    
    # Create deployment configuration
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        description="Iris species classification endpoint",
        tags={"model": "iris-classifier", "method": "sklearn"},
        enable_app_insights=True
    )
    
    # Deploy model
    service = Model.deploy(
        ws,
        service_name,
        [model],
        inference_config,
        deployment_config
    )
    
    service.wait_for_deployment(show_output=True)
    
    return service


def test_endpoint(service):
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
    result = service.run(json.dumps(test_data))
    print(f"Test result: {result}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Deploy Iris model to Azure ML')
    parser.add_argument('--config', default='config/azure-ml-config.yml')
    parser.add_argument('--model-name', default='iris-classifier')
    parser.add_argument('--service-name', default='iris-classifier-service')
    parser.add_argument('--tenant-id', help='Azure tenant ID')
    parser.add_argument('--service-principal-id', help='Service principal ID')
    parser.add_argument('--service-principal-password', help='Service principal password')
    parser.add_argument('--test-endpoint', action='store_true', help='Test the endpoint after deployment')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Authenticate
        auth = authenticate_azure_ml(
            args.tenant_id,
            args.service_principal_id,
            args.service_principal_password
        )
        
        # Get workspace
        ws = get_workspace(config, auth)
        
        # Get latest model
        model = get_latest_model(ws, args.model_name)
        if model is None:
            print("No trained model found. Please train a model first.")
            return
        
        # Create inference configuration
        inference_config = create_inference_config()
        
        # Deploy model
        service = deploy_model(ws, model, inference_config, args.service_name)
        
        print(f"Model deployed successfully!")
        print(f"Service name: {service.name}")
        print(f"Scoring URI: {service.scoring_uri}")
        
        if args.test_endpoint:
            test_endpoint(service)
        
        # Save deployment info
        deployment_info = {
            "service_name": service.name,
            "scoring_uri": service.scoring_uri,
            "state": service.state,
            "deployed_at": service.created_time.isoformat() if service.created_time else None
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