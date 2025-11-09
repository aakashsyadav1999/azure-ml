"""
Test script for the deployed Iris classification endpoint
"""

import json
import requests
import argparse


def test_endpoint(scoring_uri, auth_key=None):
    """Test the deployed endpoint with sample data"""
    
    # Sample Iris data for testing (sepal_length, sepal_width, petal_length, petal_width)
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Should be setosa
        [6.0, 3.0, 4.8, 1.8],  # Should be virginica  
        [5.8, 2.7, 4.1, 1.0],  # Should be versicolor
        [4.9, 3.0, 1.4, 0.2],  # Should be setosa
        [6.7, 3.1, 4.7, 1.5],  # Should be versicolor
        [7.2, 3.6, 6.1, 2.5]   # Should be virginica
    ]
    
    # Expected classes
    expected_classes = ["setosa", "virginica", "versicolor", "setosa", "versicolor", "virginica"]
    class_names = ["setosa", "versicolor", "virginica"]
    
    # Prepare request data
    data = {"data": test_samples}
    
    # Set headers
    headers = {"Content-Type": "application/json"}
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"
    
    try:
        # Make request
        print(f"Testing endpoint: {scoring_uri}")
        print(f"Sending {len(test_samples)} samples...")
        
        response = requests.post(scoring_uri, json=data, headers=headers)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if "error" in result:
            print(f"❌ Error from endpoint: {result['error']}")
            return False
            
        predictions = result["predictions"]
        probabilities = result["probabilities"]
        
        print("\n🎯 **Test Results:**")
        print("-" * 60)
        
        correct_predictions = 0
        for i, (sample, pred_idx, probs, expected) in enumerate(zip(test_samples, predictions, probabilities, expected_classes)):
            predicted_class = class_names[pred_idx]
            confidence = max(probs) * 100
            is_correct = predicted_class == expected
            
            if is_correct:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} Sample {i+1}: {sample}")
            print(f"   Predicted: {predicted_class} (confidence: {confidence:.1f}%)")
            print(f"   Expected:  {expected}")
            print(f"   Probabilities: {[f'{p:.3f}' for p in probs]}")
            print()
        
        accuracy = (correct_predictions / len(test_samples)) * 100
        print(f"🎯 **Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_samples)})**")
        
        if accuracy >= 80:
            print("🎉 **Endpoint is working well!**")
        else:
            print("⚠️  **Endpoint accuracy could be better**")
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False


def get_endpoint_info():
    """Load endpoint info from deployment file"""
    try:
        with open("deployment/deployment_info.json", 'r') as f:
            info = json.load(f)
        return info.get("scoring_uri")
    except FileNotFoundError:
        print("❌ deployment_info.json not found. Run deployment first.")
        return None
    except Exception as e:
        print(f"❌ Error loading deployment info: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test Iris classification endpoint')
    parser.add_argument('--endpoint-url', help='Scoring URI of the endpoint')
    parser.add_argument('--auth-key', help='Authentication key (if required)')
    
    args = parser.parse_args()
    
    # Get endpoint URL
    if args.endpoint_url:
        scoring_uri = args.endpoint_url
    else:
        scoring_uri = get_endpoint_info()
        
    if not scoring_uri:
        print("❌ No endpoint URL provided. Use --endpoint-url or ensure deployment_info.json exists.")
        return
    
    # Test the endpoint
    success = test_endpoint(scoring_uri, args.auth_key)
    
    if success:
        print("\n✨ **Endpoint testing completed successfully!**")
    else:
        print("\n❌ **Endpoint testing failed**")


if __name__ == "__main__":
    main()