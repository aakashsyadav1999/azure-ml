#!/bin/bash

echo "🚀 Azure Resource Provider Registration Script"
echo "============================================="
echo ""
echo "This script registers the required Azure resource providers for Azure ML deployments."
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first:"
    echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    echo "❌ Not logged in to Azure. Please login first:"
    echo "   az login"
    exit 1
fi

echo "✅ Azure CLI is installed and you're logged in."
echo ""

# Get current subscription info
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

echo "📋 Current subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"
echo ""

# Resource providers to register
declare -a providers=(
    "Microsoft.MachineLearningServices"
    "Microsoft.ContainerInstance"
    "Microsoft.ContainerRegistry"
    "Microsoft.KeyVault"
    "Microsoft.Storage"
    "Microsoft.Compute"
)

echo "🔧 Registering resource providers..."
echo ""

for provider in "${providers[@]}"; do
    echo "📦 Registering $provider..."
    
    # Check current status
    STATUS=$(az provider show --namespace $provider --query registrationState -o tsv 2>/dev/null || echo "NotFound")
    
    if [ "$STATUS" = "Registered" ]; then
        echo "   ✅ Already registered"
    else
        echo "   ⏳ Current status: $STATUS"
        echo "   🔄 Registering..."
        
        if az provider register --namespace $provider --output none; then
            echo "   ✅ Registration initiated"
        else
            echo "   ❌ Failed to register $provider"
        fi
    fi
    echo ""
done

echo "⏱️  Waiting for registrations to complete..."
echo "   This can take 5-10 minutes..."
echo ""

# Wait for all providers to be registered
all_registered=false
max_attempts=30
attempt=1

while [ "$all_registered" = false ] && [ $attempt -le $max_attempts ]; do
    echo "🔍 Check $attempt/$max_attempts..."
    all_registered=true
    
    for provider in "${providers[@]}"; do
        STATUS=$(az provider show --namespace $provider --query registrationState -o tsv 2>/dev/null || echo "NotFound")
        
        if [ "$STATUS" != "Registered" ]; then
            echo "   ⏳ $provider: $STATUS"
            all_registered=false
        else
            echo "   ✅ $provider: Registered"
        fi
    done
    
    if [ "$all_registered" = false ]; then
        echo "   Waiting 30 seconds before next check..."
        sleep 30
        ((attempt++))
    fi
    echo ""
done

if [ "$all_registered" = true ]; then
    echo "🎉 All resource providers are now registered!"
    echo ""
    echo "✅ You can now deploy Azure ML managed online endpoints."
    echo "   Run your deployment script again: python src/deploy_model.py"
else
    echo "⚠️  Some resource providers are still registering."
    echo "   You can check the status manually with:"
    echo "   az provider list --query \"[?namespace=='Microsoft.MachineLearningServices']\""
    echo ""
    echo "   Or continue waiting - registration usually completes within 10-15 minutes."
fi

echo ""
echo "🔗 Useful links:"
echo "   - Azure Resource Providers: https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/resource-providers-and-types"
echo "   - Azure ML troubleshooting: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-deployment"