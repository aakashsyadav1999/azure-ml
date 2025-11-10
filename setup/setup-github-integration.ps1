# GitHub Secrets Setup Script
# This script helps configure the required secrets for GitHub Actions

# Variables - Replace these with your actual values
$subscriptionId = "YOUR_SUBSCRIPTION_ID"
$resourceGroupName = "YOUR_RESOURCE_GROUP"
$workspaceName = "YOUR_WORKSPACE_NAME"
$appName = "iris-mlops-github-app"

Write-Host "🔧 Setting up GitHub integration for Azure ML" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

# 1. Register Azure AD application
Write-Host "`n1️⃣ Creating Azure AD Application..." -ForegroundColor Yellow
$app = az ad app create --display-name $appName --query "{appId:appId, objectId:id}" -o json | ConvertFrom-Json

$appId = $app.appId
$objectId = $app.objectId

Write-Host "   ✅ App ID: $appId" -ForegroundColor Green
Write-Host "   ✅ Object ID: $objectId" -ForegroundColor Green

# 2. Create Service Principal
Write-Host "`n2️⃣ Creating Service Principal..." -ForegroundColor Yellow
$sp = az ad sp create --id $appId --query "{objectId:id}" -o json | ConvertFrom-Json
$spObjectId = $sp.objectId

Write-Host "   ✅ Service Principal Object ID: $spObjectId" -ForegroundColor Green

# 3. Get Tenant ID
Write-Host "`n3️⃣ Getting Tenant ID..." -ForegroundColor Yellow
$tenantId = az account show --query "tenantId" -o tsv
Write-Host "   ✅ Tenant ID: $tenantId" -ForegroundColor Green

# 4. Create Role Assignment
Write-Host "`n4️⃣ Creating Role Assignment..." -ForegroundColor Yellow
az role assignment create --role "Contributor" --assignee $spObjectId --scope "/subscriptions/$subscriptionId/resourceGroups/$resourceGroupName"
Write-Host "   ✅ Role assignment created" -ForegroundColor Green

# 5. Display required GitHub Secrets
Write-Host "`n🔑 GITHUB SECRETS TO CONFIGURE" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "Go to your GitHub repository → Settings → Secrets and variables → Actions" -ForegroundColor White
Write-Host "`nCreate the following secrets:" -ForegroundColor White
Write-Host "AZURE_CLIENT_ID: $appId" -ForegroundColor Yellow
Write-Host "AZURE_TENANT_ID: $tenantId" -ForegroundColor Yellow  
Write-Host "AZURE_SUBSCRIPTION_ID: $subscriptionId" -ForegroundColor Yellow
Write-Host "AZURE_RESOURCE_GROUP: $resourceGroupName" -ForegroundColor Yellow
Write-Host "AZURE_WORKSPACE_NAME: $workspaceName" -ForegroundColor Yellow

Write-Host "`n🌍 GITHUB ENVIRONMENTS TO CREATE" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Go to your GitHub repository → Settings → Environments" -ForegroundColor White
Write-Host "Create these environments with the above secrets:" -ForegroundColor White
Write-Host "- Development" -ForegroundColor Yellow
Write-Host "- Production" -ForegroundColor Yellow

Write-Host "`n✅ Setup completed! Configure the secrets in GitHub and you're ready to go!" -ForegroundColor Green