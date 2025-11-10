# Professional MLOps with Iris Classification

This repository demonstrates enterprise-grade Machine Learning Operations (MLOps) practices using Azure Machine Learning and GitHub Actions, following industry best practices and patterns.

## 🏗️ Architecture Overview

```
azure_ml_trial/
├── Pipeline/                    # Main ML pipeline orchestration
│   ├── deploy-train.py         # Training pipeline orchestrator
│   ├── components/             # Reusable ML components
│   └── scripts/               # Individual pipeline scripts
├── .github/workflows/          # CI/CD workflows
│   ├── train_model_dev.yml    # Development environment
│   └── train_model_prod.yml   # Production environment
├── environments/               # Environment configurations
├── docs/                      # Documentation
├── setup/                     # Setup scripts and templates
└── data/                     # Dataset storage
```

## 🎯 Key Features

- **🔄 Component-Based Pipeline**: Modular, reusable ML components
- **🌍 Environment Separation**: Dev/Prod workflows with proper governance
- **📊 MLflow Integration**: Comprehensive experiment tracking and model comparison
- **🤖 Automated CI/CD**: GitHub Actions for seamless deployments
- **🔐 Security Best Practices**: Azure identity management and secret handling
- **📈 Model Registry**: Automatic model versioning and registration
- **🧪 A/B Testing Ready**: Multiple model comparison and selection

## 🚀 Quick Start

### Prerequisites

- Azure subscription with Azure ML workspace
- GitHub repository with Actions enabled
- Azure CLI installed locally
- Python 3.9+

### 1. Setup Azure Integration

```bash
# Clone the repository
git clone <your-repo-url>
cd azure_ml_trial

# Run the setup script
cd setup
./setup-github-integration.ps1
```

### 2. Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Create these secrets:
- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID` 
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP`
- `AZURE_WORKSPACE_NAME`

### 3. Create GitHub Environments

Go to Settings → Environments and create:
- **Development**: For dev branch deployments
- **Production**: For main/master branch deployments

### 4. Run Your First Pipeline

```bash
# Push to dev branch to trigger development pipeline
git checkout -b dev
git push origin dev

# Or run manually
python Pipeline/deploy-train.py --model-name iris-classifier-v1
```

## 🏭 Pipeline Components

### 1. Data Preparation (`data_preparation.py`)
- Loads and splits the Iris dataset
- Creates train/test splits with stratification
- Outputs structured CSV files

### 2. Feature Engineering (`feature_engineering.py`) 
- Creates interaction features (ratios, areas)
- Generates categorical size features
- Applies domain-specific transformations

### 3. Model Training (`train_model.py`)
- Trains multiple model types (Random Forest, Logistic Regression, SVM)
- Compares performance with MLflow tracking
- Selects best model based on accuracy
- Comprehensive metrics logging

### 4. Model Registration (`register_model.py`)
- Registers best model to Azure ML Model Registry
- Creates model metadata and versioning
- Tags models with performance metrics

## 🔧 Configuration

### Environment Configuration
- `environments/dev.yml`: Development settings
- `environments/prod.yml`: Production settings
- `environments/iris-training-env.yml`: Conda dependencies

### Pipeline Settings
- Component definitions in `Pipeline/components/`
- Orchestration logic in `Pipeline/deploy-train.py`
- Environment-specific compute configurations

## 📊 MLflow Integration

All training runs include comprehensive tracking:

- **Parameters**: Model hyperparameters, data splits
- **Metrics**: Accuracy, F1-score, precision, recall
- **Artifacts**: Trained models, classification reports
- **Model Comparison**: Side-by-side performance analysis

Access MLflow UI through Azure ML Studio → Experiments

## 🔄 CI/CD Workflows

### Development Workflow (`train_model_dev.yml`)
- Triggers on `dev` branch pushes
- Uses Development environment
- Fast feedback for experimentation

### Production Workflow (`train_model_prod.yml`) 
- Triggers on `main`/`master` branch pushes
- Uses Production environment  
- Includes additional validation and governance

## 🧪 Local Development

### Setup Local Environment

```bash
# Create conda environment
conda env create -f environments/iris-training-env.yml
conda activate iris-training-env

# Install additional dependencies
pip install -r requirements.txt
```

### Run Components Locally

```bash
# Data preparation
python Pipeline/scripts/data_preparation.py \
  --output_train_data ./local_outputs/train \
  --output_test_data ./local_outputs/test

# Feature engineering  
python Pipeline/scripts/feature_engineering.py \
  --input_train_data ./local_outputs/train \
  --input_test_data ./local_outputs/test \
  --output_train_data ./local_outputs/processed_train \
  --output_test_data ./local_outputs/processed_test

# Model training
python Pipeline/scripts/train_model.py \
  --input_train_data ./local_outputs/processed_train \
  --input_test_data ./local_outputs/processed_test \
  --output_model_dir ./local_outputs/models \
  --output_metrics_dir ./local_outputs/metrics
```

### Run Full Pipeline

```bash
# Create Azure ML config
mkdir -p .azureml
cp setup/config.json.template .azureml/config.json
# Edit .azureml/config.json with your values

# Run pipeline
python Pipeline/deploy-train.py \
  --model-name iris-classifier-local \
  --experiment-name iris-classification-dev \
  --wait-for-completion
```

## 🔍 Monitoring and Debugging

### Check Job Status
- Azure ML Studio → Jobs → Your experiment
- GitHub Actions → Actions tab → Workflow runs
- Local logs in terminal output

### Common Issues
1. **Authentication Errors**: Check Azure credentials and permissions
2. **Compute Issues**: Verify compute cluster exists and has quota
3. **Environment Issues**: Check conda environment and dependencies
4. **Data Issues**: Verify data paths and file permissions

### Debug Locally
```bash
# Test Azure connection
az ml compute list

# Test components individually
python Pipeline/scripts/data_preparation.py --help

# Check MLflow tracking
mlflow ui
```

## 📈 Extending the Pipeline

### Add New Components
1. Create script in `Pipeline/scripts/`
2. Add YAML definition in `Pipeline/components/`
3. Update `Pipeline/deploy-train.py` orchestration
4. Test locally before committing

### Add New Models
1. Update `train_model.py` with new algorithms
2. Add hyperparameter configurations
3. Update model comparison logic
4. Test performance impact

### Add Deployment
1. Create `deploy-score.py` orchestrator
2. Add batch/real-time endpoint scripts
3. Create deployment workflows
4. Add monitoring and alerting

## 🔒 Security Best Practices

- ✅ Service principal authentication
- ✅ Environment-based secret management  
- ✅ Least privilege access principles
- ✅ Audit trail through Git and Azure logs
- ✅ No credentials in code or configs

## 📚 Additional Resources

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [GitHub Actions Documentation](https://docs.github.com/actions)
- [MLOps Best Practices](https://ml-ops.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test locally and with dev workflow
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for the MLOps community**