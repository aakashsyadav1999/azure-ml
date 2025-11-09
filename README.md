# 🚀 Azure ML Automated Training Demo

[![Azure ML Training](https://github.com/aakashsyadav1999/azure-ml/actions/workflows/azure-ml-training.yml/badge.svg)](https://github.com/aakashsyadav1999/azure-ml/actions/workflows/azure-ml-training.yml)

A complete demonstration of automated machine learning model training using **Azure ML** and **GitHub Actions**. This project shows how a simple commit to the `dev` branch automatically triggers ML model training in the cloud.

## 🎯 Demo Overview

```
📝 Git Commit → 🔄 GitHub Actions → ☁️ Azure ML Training → 🤖 Model Saved
```

This project demonstrates:
- **Automated ML Pipeline**: Train models automatically when code changes
- **Cloud Integration**: Seamless Azure ML workspace integration
- **CI/CD for ML**: Modern MLOps practices with GitHub Actions
- **Iris Classification**: Classic ML problem with 95%+ accuracy results

## 🏗️ **Project Structure**

```
azure-ml-trial/
├── � src/                          # Source code
│   ├── � generate_data.py         # Iris dataset generation
│   ├── 📄 train_model.py           # Model training script  
│   ├── � azure_ml_pipeline.py     # Azure ML job submission
│   └── 📄 deploy_model.py          # Model deployment script
├── � config/                      # Configuration files
│   ├── 📄 azure-ml-config.yml      # Azure ML workspace config
│   └── 📄 conda-env.yml            # Conda environment
├── 📂 .github/workflows/           # CI/CD automation
│   └── 📄 azure-ml-training.yml    # GitHub Actions workflow
├── � data/                        # Datasets (auto-generated)
│   ├── 📄 train.csv               # Training data
│   └── 📄 test.csv                # Test data
├── 📂 models/                      # Saved models
│   ├── 📄 latest_model.joblib     # Latest trained model
│   └── 📄 latest_metadata.json    # Model metadata
├── 📂 deployment/                  # Deployment artifacts
│   ├── 📄 score.py                # Scoring script (auto-generated)
│   ├── 📄 conda-env.yml           # Inference environment
│   └── 📄 deployment_info.json    # Endpoint information
├── 📄 test_endpoint.py             # Endpoint testing script
├── � requirements.txt             # Python dependencies
├── 📄 main.py                      # Entry point for demo
├── � README.md                    # This file
└── 📄 DEPLOYMENT.md                # Deployment guide
```

## 🎮 Quick Start Demo

### Option 1: Trigger Full Pipeline (Training + Deployment)
```bash
# Make any change to trigger training
echo "Demo update $(date)" >> README.md

# Commit and push to dev branch
git add .
git commit -m "Trigger Azure ML training and deployment"
git push origin dev

# 🎉 Watch GitHub Actions → Azure ML Studio for training and deployment!
```

### Option 2: Manual Workflow Trigger
1. Go to [GitHub Actions](https://github.com/aakashsyadav1999/azure-ml/actions)
2. Click "Azure ML Automated Training Pipeline"
3. Click "Run workflow" → Select `dev` branch → "Run workflow"

### Option 3: Test Deployed Endpoint
```bash
# Test your deployed model endpoint
python test_endpoint.py --endpoint-url YOUR_SCORING_URI

# Or check deployment_info.json for endpoint details
cat deployment/deployment_info.json
```

## 🛠️ Local Development

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Setup & Test Locally
```bash
# Clone repository
git clone https://github.com/aakashsyadav1999/azure-ml.git
cd azure_ml_trial

# Install dependencies
uv add -r requirements.txt

# Generate Iris dataset
uv run python src/generate_data.py --output-dir data

# Train model locally
uv run python src/train_model.py \
  --data-path data/train.csv \
  --model-type random_forest \
  --output-dir models

# Check results
ls data/      # train.csv, test.csv
ls models/    # latest_model.joblib, latest_metadata.json
```

## ☁️ Azure Configuration

### Current Setup
- **Subscription**: `146253af-8dff-4ab7-b56f-c2bf8f847a24` (aakash-main-subscription)
- **Resource Group**: `aakash-trial`
- **Workspace**: `iris-dataset`
- **Region**: `central india`
- **Compute**: `ml-training-cluster` (Standard_DS3_v2)

### GitHub Secrets (Pre-configured)
| Secret Name | Description |
|-------------|-------------|
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_CLIENT_ID` | Service principal app ID |
| `AZURE_CLIENT_SECRET` | Service principal secret |
| `AZURE_RESOURCE_GROUP` | Resource group name |
| `AZURE_WORKSPACE_NAME` | Azure ML workspace name |

## 🤖 ML Pipeline Details

### Dataset: Iris Classification
- **Source**: scikit-learn built-in dataset
- **Samples**: 150 total (120 train, 30 test)
- **Features**: 4 (sepal length/width, petal length/width)
- **Classes**: 3 (setosa, versicolor, virginica)

### Models Supported
- **Random Forest** (default): Ensemble method, 100 estimators
- **Logistic Regression**: Linear classification with regularization

### Training Metrics
- **Accuracy**: ~95.8% validation accuracy
- **F1-Score**: ~95.8% macro-averaged F1
- **Training Time**: ~30 seconds on Azure ML compute

### Sample Output
```
Training random_forest model...
Loaded data: 120 samples, 4 features
Training Accuracy: 1.0000
Validation Accuracy: 0.9583
Validation F1: 0.9582

Validation classification report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         8
           1       1.00      0.88      0.93         8
           2       0.89      1.00      0.94         8
```

## 🔄 Workflow Process

### 1. Data Generation Job
```yaml
- Generate Iris dataset from scikit-learn
- Split into train/test CSV files
- Upload as artifacts for training job
```

### 2. Azure ML Training Job
```yaml
- Authenticate with service principal
- Install Azure ML SDK and dependencies
- Submit training job to Azure ML workspace
- Monitor job execution
```

### 3. Training Execution (Azure ML)
```yaml
- Create/use compute cluster (ml-training-cluster)
- Set up conda environment with ML dependencies
- Run train_model.py with Iris data
- Save trained model and metadata
- Register model in Azure ML Model Registry
```

### 4. Model Deployment (NEW!)
```yaml
- Create inference configuration with scoring script
- Deploy model to Azure Container Instance (ACI)
- Test endpoint with sample Iris predictions
- Generate deployment info and monitoring links
```

## 🚀 **Complete MLOps Pipeline**

### **🎯 What Happens When You Commit?**

**Full MLOps Pipeline Triggered by `git push origin dev`:**

1. **🔥 Data Preparation**
   - Loads Iris dataset (150 samples, 4 features, 3 classes)
   - Creates stratified train/test splits
   - Uploads data as GitHub Actions artifacts

2. **🤖 Model Training on Azure ML**
   - Submits job to Azure ML compute cluster
   - Trains RandomForest classifier (95.8% accuracy achieved!)
   - Saves model artifacts and metadata
   - **🆕 Automatically registers model in Azure ML Model Registry**

3. **🚀 Model Deployment (NEW!)**
   - Creates inference configuration with scoring script
   - Deploys model to Azure Container Instance (ACI) endpoint
   - Tests endpoint with sample predictions
   - Saves deployment info for monitoring

4. **✅ Notification**
   - Sends status notifications about training and deployment
   - Provides links to Azure ML Studio and endpoint details

## 📊 Monitoring & Results

### GitHub Actions
- **Workflow URL**: [Actions Tab](https://github.com/aakashsyadav1999/azure-ml/actions)
- **Status Badge**: Shows current build status
- **Logs**: Detailed execution logs for debugging

### Azure ML Studio
- **Workspace URL**: [ml.azure.com](https://ml.azure.com/)
- **Experiments**: `iris-classification` experiment
- **Models**: Trained models with versioning and registry
- **Endpoints**: Deployed web services for inference
- **Compute**: Real-time compute cluster status

### Artifacts Generated
- **Model File**: `latest_model.joblib` (RandomForest/LogisticRegression)
- **Metadata**: `latest_metadata.json` (metrics, timestamps, config)
- **Endpoint**: Live REST API for Iris species prediction
- **Deployment Info**: `deployment/deployment_info.json` with scoring URI

### Example Endpoint Usage
```bash
curl -X POST "YOUR_SCORING_URI" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [5.1, 3.5, 1.4, 0.2],
      [6.0, 3.0, 4.8, 1.8]
    ]
  }'

# Response:
{
  "predictions": [0, 2],
  "probabilities": [[0.95, 0.03, 0.02], [0.01, 0.02, 0.97]],
  "class_names": ["setosa", "versicolor", "virginica"]
}
```
- **Versioned Files**: Timestamped model and metadata files

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **"No subscriptions found"** | Check service principal permissions on subscription |
| **"Workspace not found"** | Verify `azure-ml-config.yml` has correct workspace details |
| **Import errors** | Ensure dependencies are installed consistently (pip vs uv) |
| **Authentication failed** | Check GitHub secrets are set correctly |
| **Compute creation timeout** | Standard_DS3_v2 VMs may take 5-10 minutes to provision |

### Debug Commands
```bash
# Test data generation locally
uv run python src/generate_data.py --output-dir data
head data/train.csv

# Test model training locally
uv run python src/train_model.py --data-path data/train.csv --output-dir models

# Check model metadata
cat models/latest_metadata.json
```

## 🔧 Customization

### Change Model Type
```bash
# Train with different algorithms
uv run python src/train_model.py \
  --data-path data/train.csv \
  --model-type logistic_regression \
  --output-dir models
```

### Modify Data Split
```bash
# Generate different train/test split
uv run python src/generate_data.py \
  --output-dir data \
  --test-size 0.3 \
  --random-state 123
```

### Update Azure ML Config
Edit `config/azure-ml-config.yml`:
```yaml
compute:
  vm_size: "Standard_DS2_v2"  # Smaller/larger VM
  max_instances: 2            # Scale up compute
```

## 🎤 Presentation Demo Script

1. **Show the repository structure**
   > "This is a complete MLOps pipeline that trains ML models automatically"

2. **Demonstrate the trigger**
   > "Watch what happens when I commit to the dev branch..."
   ```bash
   echo "Demo for audience $(date)" >> README.md
   git add . && git commit -m "Live demo" && git push origin dev
   ```

3. **Show GitHub Actions**
   > "GitHub Actions immediately picks up the change and starts the workflow"

4. **Show Azure ML Studio**
   > "The training job is now running in Azure ML cloud compute"

5. **Show Results**
   > "Within minutes, we have a trained model with 95%+ accuracy!"

## 📚 Learn More

- [Azure Machine Learning Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [GitHub Actions for Azure](https://docs.github.com/actions/deployment/deploying-to-azure)
- [MLOps with Azure ML](https://docs.microsoft.com/azure/machine-learning/concept-model-management-and-deployment)
- [Iris Dataset Background](https://en.wikipedia.org/wiki/Iris_flower_data_set)

## 🤝 Contributing

This is a demo project, but improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -m 'Add amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🏆 Acknowledgments

- **Iris Dataset**: Ronald Fisher's classic 1936 dataset
- **Azure ML**: Microsoft's cloud ML platform
- **GitHub Actions**: Automated CI/CD workflows
- **scikit-learn**: Python machine learning library

---

**🎯 Demo Status: ✅ Production Ready**

*Last Updated: November 2025*
