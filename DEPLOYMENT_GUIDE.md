# 🚀 Model Deployment Guide

This guide shows you how to deploy your trained and registered iris-classifier model when you're ready.

## Prerequisites

✅ Model successfully trained and registered in Azure ML Model Registry
✅ Azure CLI installed and logged in
✅ Sufficient Azure quota for compute resources

## 🎯 Deployment Options

### Option 1: Azure CLI (Recommended)

```bash
# 1. Create endpoint configuration
cat > endpoint.yml << EOF
name: iris-classifier-endpoint
description: Iris species classification endpoint
auth_mode: key
tags:
  model: iris-classifier
  environment: production
EOF

# 2. Create deployment configuration  
cat > deployment.yml << EOF
name: iris-deployment
endpoint_name: iris-classifier-endpoint
model: azureml:iris-classifier:latest
environment: azureml:iris-training-env:1
instance_type: Standard_DS1_v2
instance_count: 1
EOF

# 3. Deploy
az ml online-endpoint create --file endpoint.yml
az ml online-deployment create --file deployment.yml
az ml online-endpoint update --name iris-classifier-endpoint --traffic "iris-deployment=100"
```

### Option 2: Azure ML Studio (GUI)

1. **Navigate to Azure ML Studio**: https://ml.azure.com
2. **Go to Models** → Select `iris-classifier` → Latest version
3. **Click "Deploy"** → Choose "Real-time endpoint"
4. **Configure deployment**:
   - Endpoint name: `iris-classifier-endpoint`
   - Compute type: `Managed online endpoint`
   - Instance type: `Standard_DS1_v2` (or smaller for quota efficiency)
   - Instance count: `1`
5. **Deploy** and wait for completion

## 🧪 Testing Your Deployed Model

### Test with Sample Data

```python
import requests
import json

# Get endpoint URL and key from Azure ML Studio
endpoint_url = "https://iris-classifier-endpoint.region.inference.ml.azure.com/score"
api_key = "your-endpoint-key"

# Sample iris data [sepal_length, sepal_width, petal_length, petal_width]
test_data = {
    "data": [
        [5.1, 3.5, 1.4, 0.2],  # Should predict Setosa
        [6.2, 2.9, 4.3, 1.3],  # Should predict Versicolor  
        [7.3, 2.9, 6.3, 1.8]   # Should predict Virginica
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post(endpoint_url, json=test_data, headers=headers)
print(f"Predictions: {response.json()}")
```

### Use the Test Script

```bash
python test_endpoint.py \
  --endpoint-url "your-endpoint-url" \
  --api-key "your-endpoint-key"
```

## 🔧 Troubleshooting Deployment

### Common Issues

1. **Quota Limits**: Use smaller VM sizes (`Standard_B1s`, `Standard_DS1_v2`)
2. **Resource Providers**: Run `./scripts/register-azure-providers.sh`
3. **Failed Endpoints**: Run `./scripts/cleanup-endpoints.sh`

### Monitoring

- **Azure ML Studio**: Real-time metrics and logs
- **Application Insights**: Detailed telemetry (if configured)
- **Azure Monitor**: Resource utilization

## 🧹 Cleanup

```bash
# Delete endpoint when no longer needed
az ml online-endpoint delete --name iris-classifier-endpoint

# Or use the cleanup script
./scripts/cleanup-endpoints.sh
```

## 💡 Deployment Best Practices

1. **Start Small**: Use minimal compute for testing, scale up as needed
2. **Monitor Costs**: Set up billing alerts for endpoint usage
3. **Version Control**: Deploy specific model versions, not "latest"
4. **Health Checks**: Implement readiness and liveness probes
5. **Security**: Use managed identity instead of API keys when possible

## 📚 Additional Resources

- [Azure ML Managed Online Endpoints](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-managed-online-endpoints)
- [Deployment Troubleshooting](https://docs.microsoft.com/azure/machine-learning/how-to-troubleshoot-deployment)
- [Endpoint Security](https://docs.microsoft.com/azure/machine-learning/how-to-authenticate-web-service)