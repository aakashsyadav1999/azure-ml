# GitHub Repository Secrets Setup

## Required Secrets for Azure ML Pipeline

To enable GitHub Actions to run your Azure ML training pipeline, you need to configure the following secrets in your repository:

### Authentication Secrets
- `AZURE_CLIENT_ID`: Service principal client ID
- `AZURE_TENANT_ID`: Your Azure tenant ID  
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID

### Workspace Configuration Secrets
- `AZURE_RESOURCE_GROUP`: `aakash-trial`
- `AZURE_WORKSPACE_NAME`: `iris-data`

## How to Update Secrets

1. Go to your repository: https://github.com/aakashsyadav1999/azure-ml
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"** or edit existing ones
4. Add/update each secret with the correct values

## Current Workspace Information (from Azure Portal)
- **Subscription**: aakash-main-subscription
- **Workspace**: iris-data
- **Resource Group**: aakash-trial  
- **Subscription ID**: 146253af-8dff-4ab7-b56f-c2bf8f847a24
- **Location**: centralindia

## Troubleshooting

If you see error messages like:
```
The Resource 'Microsoft.MachineLearningServices/workspaces/iris-dataset' under resource group 'aakash-trial' was not found
```

This means the `AZURE_WORKSPACE_NAME` secret is still set to the old value `iris-dataset`. Update it to `iris-data`.

## Service Principal Setup

If you don't have a service principal yet, create one:

```bash
# Create service principal
az ad sp create-for-rbac --name "azure-ml-pipeline-sp" \
                         --role contributor \
                         --scopes /subscriptions/146253af-8dff-4ab7-b56f-c2bf8f847a24/resourceGroups/aakash-trial

# This will output:
# {
#   "appId": "your-client-id",        # Use this for AZURE_CLIENT_ID
#   "displayName": "azure-ml-pipeline-sp",
#   "password": "your-client-secret", # Use this for AZURE_CLIENT_SECRET (if needed)
#   "tenant": "your-tenant-id"        # Use this for AZURE_TENANT_ID
# }
```

## Verification

After updating secrets, trigger a new pipeline run by:
1. Making a small commit to the `dev` branch
2. Or manually triggering via **Actions** → **DEV - Train Model** → **Run workflow**