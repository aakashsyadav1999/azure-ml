# Fix Azure Authentication for GitHub Actions

## Current Issue
Your service principal `ml-training-sp` needs federated identity credentials configured for GitHub Actions OIDC authentication.

## Option A: Configure Federated Identity Credentials (RECOMMENDED)

### Step 1: Configure Federated Identity via Azure CLI

```bash
# Login to Azure CLI first
az login

# Get your service principal details
az ad sp list --display-name "ml-training-sp" --query "[].{appId:appId,objectId:id}" --output table

# Configure federated identity credential for Development environment
az ad app federated-credential create \
  --id <YOUR_APP_ID_FROM_ABOVE> \
  --parameters '{
    "name": "github-dev-environment",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:aakashsyadav1999/azure-ml:environment:Development",
    "description": "GitHub Actions - Development Environment",
    "audiences": [
      "api://AzureADTokenExchange"
    ]
  }'

# Configure federated identity credential for Production environment  
az ad app federated-credential create \
  --id <YOUR_APP_ID_FROM_ABOVE> \
  --parameters '{
    "name": "github-prod-environment",
    "issuer": "https://token.actions.githubusercontent.com", 
    "subject": "repo:aakashsyadav1999/azure-ml:environment:Production",
    "description": "GitHub Actions - Production Environment",
    "audiences": [
      "api://AzureADTokenExchange"
    ]
  }'
```

### Step 2: Verify Current GitHub Secrets

Make sure you have these secrets configured in GitHub:
- `AZURE_CLIENT_ID`: Your service principal app ID
- `AZURE_TENANT_ID`: Your Azure tenant ID  
- `AZURE_SUBSCRIPTION_ID`: 146253af-8dff-4ab7-b56f-c2bf8f847a24
- `AZURE_WORKSPACE_NAME`: iris-data
- `AZURE_RESOURCE_GROUP`: aakash-trial

## Option B: Use Service Principal Secret (Less Secure)

If you prefer the older method with client secrets:

### Step 1: Get/Create Client Secret

```bash
# Create a new client secret for your service principal
az ad app credential reset --id <YOUR_APP_ID> --display-name "GitHub Actions Secret"
```

### Step 2: Update GitHub Workflow

Update `.github/workflows/train_model_dev.yml` to use client secret instead of OIDC:

```yaml
    - name: Azure Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
```

### Step 3: Add AZURE_CREDENTIALS Secret

Create a secret named `AZURE_CREDENTIALS` with this JSON format:
```json
{
  "clientId": "<your-client-id>",
  "clientSecret": "<your-client-secret>",
  "subscriptionId": "146253af-8dff-4ab7-b56f-c2bf8f847a24",
  "tenantId": "<your-tenant-id>"
}
```

## Recommended Solution

**Use Option A** (Federated Identity) as it's more secure and doesn't require managing client secrets.

After configuring federated identity credentials, your GitHub Actions will authenticate securely without any secrets!