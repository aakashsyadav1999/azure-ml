#!/bin/bash

echo "🧹 Azure ML Endpoint Cleanup Script"
echo "===================================="
echo ""

# Check if Azure CLI is installed and logged in
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first:"
    echo "   https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

if ! az account show &> /dev/null; then
    echo "❌ Not logged in to Azure. Please login first:"
    echo "   az login"
    exit 1
fi

# Install ML extension if not present
if ! az extension list | grep -q ml; then
    echo "📦 Installing Azure ML CLI extension..."
    az extension add -n ml
fi

echo "✅ Azure CLI is ready"
echo ""

# Get current subscription info
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
echo "📋 Current subscription: $SUBSCRIPTION_NAME"
echo ""

# List Azure ML workspaces
echo "🔍 Finding Azure ML workspaces..."
WORKSPACES=$(az ml workspace list --query "[].{name:name, resourceGroup:resourceGroup}" -o tsv)

if [ -z "$WORKSPACES" ]; then
    echo "❌ No Azure ML workspaces found in current subscription."
    exit 1
fi

echo "🏢 Available workspaces:"
echo "$WORKSPACES" | while read -r line; do
    WS_NAME=$(echo $line | cut -f1)
    RG_NAME=$(echo $line | cut -f2)
    echo "   • $WS_NAME (Resource Group: $RG_NAME)"
done
echo ""

# Ask user to select workspace or provide default
read -p "Enter workspace name (or press Enter for 'iris-data'): " USER_WORKSPACE
WORKSPACE_NAME=${USER_WORKSPACE:-iris-data}

echo "🎯 Using workspace: $WORKSPACE_NAME"
echo ""

# List online endpoints
echo "🔍 Checking online endpoints in workspace '$WORKSPACE_NAME'..."

ENDPOINTS=$(az ml online-endpoint list --workspace-name "$WORKSPACE_NAME" --query "[].{name:name, state:provisioning_state, traffic:traffic}" -o tsv 2>/dev/null)

if [ -z "$ENDPOINTS" ]; then
    echo "✅ No online endpoints found. Nothing to clean up!"
    exit 0
fi

echo "📋 Current online endpoints:"
echo ""
echo "Name                     State        Traffic"
echo "------------------------|-----------|---------"
echo "$ENDPOINTS" | while read -r line; do
    EP_NAME=$(echo $line | cut -f1)
    EP_STATE=$(echo $line | cut -f2)
    EP_TRAFFIC=$(echo $line | cut -f3)
    
    # Color coding for states
    if [[ "$EP_STATE" == "Succeeded" ]]; then
        STATE_COLOR="✅ $EP_STATE"
    elif [[ "$EP_STATE" == "Failed" || "$EP_STATE" == "Deleting" ]]; then
        STATE_COLOR="❌ $EP_STATE"
    else
        STATE_COLOR="⏳ $EP_STATE"
    fi
    
    printf "%-24s | %-9s | %s\n" "$EP_NAME" "$STATE_COLOR" "$EP_TRAFFIC"
done
echo ""

# Find problematic endpoints
FAILED_ENDPOINTS=$(echo "$ENDPOINTS" | grep -E "(Failed|Deleting)" | cut -f1)

if [ -z "$FAILED_ENDPOINTS" ]; then
    echo "✅ All endpoints appear to be healthy!"
    
    # Ask if user wants to clean up any endpoint anyway
    read -p "Do you want to delete any endpoint? (y/N): " DELETE_CHOICE
    if [[ $DELETE_CHOICE =~ ^[Yy]$ ]]; then
        read -p "Enter endpoint name to delete: " ENDPOINT_TO_DELETE
        if [ ! -z "$ENDPOINT_TO_DELETE" ]; then
            echo "🗑️  Deleting endpoint: $ENDPOINT_TO_DELETE"
            az ml online-endpoint delete --name "$ENDPOINT_TO_DELETE" --workspace-name "$WORKSPACE_NAME" --yes
            echo "✅ Endpoint deletion initiated"
        fi
    fi
else
    echo "⚠️  Found problematic endpoints:"
    echo "$FAILED_ENDPOINTS"
    echo ""
    
    read -p "Delete all failed/deleting endpoints? (y/N): " AUTO_CLEANUP
    
    if [[ $AUTO_CLEANUP =~ ^[Yy]$ ]]; then
        echo "$FAILED_ENDPOINTS" | while read -r endpoint; do
            if [ ! -z "$endpoint" ]; then
                echo "🗑️  Deleting failed endpoint: $endpoint"
                az ml online-endpoint delete --name "$endpoint" --workspace-name "$WORKSPACE_NAME" --yes --no-wait
            fi
        done
        echo "✅ Cleanup initiated for all failed endpoints"
        echo "⏳ Deletions are running in background. Check Azure portal for progress."
    else
        echo "Manual cleanup options:"
        echo "$FAILED_ENDPOINTS" | while read -r endpoint; do
            if [ ! -z "$endpoint" ]; then
                echo "   az ml online-endpoint delete --name '$endpoint' --workspace-name '$WORKSPACE_NAME' --yes"
            fi
        done
    fi
fi

echo ""
echo "💡 Tips for avoiding quota issues:"
echo "   • Use smaller VM sizes (Standard_B1s, Standard_DS1_v2)"
echo "   • Delete unused endpoints regularly"
echo "   • Monitor quota usage: az vm list-usage --location <region>"
echo "   • Request quota increases when needed"
echo ""
echo "🔗 Useful links:"
echo "   • Azure ML pricing: https://azure.microsoft.com/pricing/details/machine-learning/"
echo "   • Quota management: https://docs.microsoft.com/azure/machine-learning/how-to-manage-quotas"