# 🚀 Model Deployment Guide

This guide covers how to deploy your trained Iris classification model as a web service endpoint on Azure ML.

## 📋 Prerequisites

1. ✅ **Successful Model Training**: Your model should be trained and registered in Azure ML
2. ✅ **Azure ML Workspace**: Configured with proper permissions
3. ✅ **Service Principal**: With contributor access to deploy endpoints

## 🎯 Deployment Process

### **Option 1: Automatic Deployment via GitHub Actions**

The GitHub Actions workflow automatically deploys models when you push to `main` or `dev` branches.

```bash
# Commit your changes and push to dev
git add .
git commit -m "Deploy model to endpoint"
git push origin dev
```

The workflow will:
1. ✅ Train the model
2. ✅ Register it in Azure ML  
3. ✅ Deploy to ACI endpoint
4. ✅ Test the endpoint

### **Option 2: Manual Deployment**

You can also deploy manually using the deployment script:

```bash
# Deploy model to endpoint
python src/deploy_model.py \
    --config config/azure-ml-config.yml \
    --model-name iris-classifier \
    --service-name iris-classifier-service \
    --test-endpoint \
    --tenant-id $AZURE_TENANT_ID \
    --service-principal-id $AZURE_CLIENT_ID \
    --service-principal-password $AZURE_CLIENT_SECRET
```

## 🔧 Deployment Components

### **1. Scoring Script (`deployment/score.py`)**
- Automatically generated during deployment
- Loads the registered model
- Provides `init()` and `run()` functions for inference

### **2. Environment Specification (`deployment/conda-env.yml`)**
- Defines the runtime environment
- Includes scikit-learn, numpy, joblib dependencies
- Ensures compatibility with your model

### **3. Inference Configuration**
- Combines scoring script + environment
- Configures how Azure ML serves your model

### **4. Deployment Configuration (ACI)**
- **CPU**: 1 core
- **Memory**: 1 GB
- **Auto-scaling**: Enabled
- **Application Insights**: Enabled for monitoring

## 🧪 Testing Your Endpoint

### **Automatic Testing**
The deployment script automatically tests your endpoint with sample Iris data.

### **Manual Testing**
```bash
# Test your deployed endpoint
python test_endpoint.py --endpoint-url YOUR_SCORING_URI
```

### **Sample Test Data**
```python
test_data = {
    "data": [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.0, 3.0, 4.8, 1.8],  # virginica
        [5.8, 2.7, 4.1, 1.0]   # versicolor
    ]
}
```

### **Expected Response**
```json
{
    "predictions": [0, 2, 1],
    "probabilities": [
        [0.95, 0.03, 0.02],
        [0.01, 0.02, 0.97],
        [0.05, 0.91, 0.04]
    ],
    "class_names": ["setosa", "versicolor", "virginica"]
}
```

## 📊 Monitoring Your Endpoint

### **Azure ML Studio**
1. Navigate to **Endpoints** in Azure ML Studio
2. Find your service: `iris-classifier-service`
3. Monitor:
   - ✅ Request volume
   - ✅ Response latency
   - ✅ Error rates
   - ✅ Resource utilization

### **Application Insights**
- Automatic logging enabled
- View detailed request/response logs
- Set up alerts for failures

## 🔄 Updating Your Deployment

### **New Model Version**
When you train a new model, the deployment script automatically updates the existing endpoint:

```bash
# Deploy updated model
python src/deploy_model.py \
    --config config/azure-ml-config.yml \
    --model-name iris-classifier \
    --service-name iris-classifier-service
```

### **Rolling Updates**
- Zero downtime deployment
- Automatic rollback on failure
- Version management

## 🔐 Security & Authentication

### **Service Authentication**
By default, the endpoint doesn't require authentication for testing. For production:

```python
# Add authentication key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_key}"
}
```

### **Network Security**
- Consider VNet integration for production
- IP allow-listing available
- HTTPS encryption enabled by default

## 🚀 Production Considerations

### **Scaling to AKS (Production)**
For high-volume production workloads, consider Azure Kubernetes Service:

```python
# Example AKS deployment config
from azureml.core.webservice import AksWebservice

aks_config = AksWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=4,
    autoscale_enabled=True,
    autoscale_min_replicas=2,
    autoscale_max_replicas=10
)
```

### **Performance Optimization**
- **Batch Inference**: For multiple predictions
- **Model Optimization**: Use ONNX for faster inference
- **Caching**: Implement result caching for repeated queries

### **Cost Management**
- **ACI**: Good for development/testing (~$0.10/hour)
- **AKS**: Better for production with predictable costs
- **Auto-shutdown**: Configure for dev environments

## 🔍 Troubleshooting

### **Common Issues**

1. **Deployment Timeout**
   ```bash
   # Check deployment logs
   service.get_logs()
   ```

2. **Import Errors**
   - Verify environment dependencies
   - Check conda-env.yml matches training environment

3. **Memory Issues**
   ```python
   # Increase memory allocation
   deployment_config = AciWebservice.deploy_configuration(
       cpu_cores=1,
       memory_gb=2  # Increased from 1GB
   )
   ```

4. **Model Loading Errors**
   - Ensure model is properly registered
   - Check model path in scoring script

### **Debugging Commands**
```bash
# View service details
az ml service show -n iris-classifier-service

# Check logs
az ml service get-logs -n iris-classifier-service

# Test locally before deployment
az ml model download -i iris-classifier:1 --target-dir ./test_model
```

## 📈 Next Steps

1. **✅ Monitor Performance**: Set up dashboards in Azure ML Studio
2. **✅ A/B Testing**: Deploy multiple model versions
3. **✅ CI/CD Integration**: Automated model validation before deployment
4. **✅ Data Drift Detection**: Monitor input data changes
5. **✅ Model Retraining**: Automated pipelines for model updates

## 🎉 Success Metrics

Your deployment is successful when:
- ✅ Endpoint responds with 200 status
- ✅ Predictions match expected accuracy (>95%)
- ✅ Response time < 1 second
- ✅ Monitoring dashboards show healthy metrics

Happy Deploying! 🚀