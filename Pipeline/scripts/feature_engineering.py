"""
Feature engineering component for Iris classification
Applies feature engineering and preprocessing steps
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os


def parse_args():
    parser = argparse.ArgumentParser("feature-engineering")
    parser.add_argument("--input_train_data", type=str, help="Path to training data")
    parser.add_argument("--input_test_data", type=str, help="Path to test data")
    parser.add_argument("--output_train_data", type=str, help="Path to save processed training data")
    parser.add_argument("--output_test_data", type=str, help="Path to save processed test data")
    
    return parser.parse_args()


def engineer_features(df):
    """Apply feature engineering to the dataset"""
    df_processed = df.copy()
    
    # Create interaction features
    df_processed['sepal_ratio'] = df_processed['sepal length (cm)'] / df_processed['sepal width (cm)']
    df_processed['petal_ratio'] = df_processed['petal length (cm)'] / df_processed['petal width (cm)']
    
    # Create area features
    df_processed['sepal_area'] = df_processed['sepal length (cm)'] * df_processed['sepal width (cm)']
    df_processed['petal_area'] = df_processed['petal length (cm)'] * df_processed['petal width (cm)']
    
    # Create total size feature
    df_processed['total_size'] = (df_processed['sepal_area'] + df_processed['petal_area'])
    
    # Create categorical features based on size
    df_processed['sepal_size_category'] = pd.cut(
        df_processed['sepal_area'], 
        bins=3, 
        labels=['Small', 'Medium', 'Large']
    )
    
    df_processed['petal_size_category'] = pd.cut(
        df_processed['petal_area'], 
        bins=3, 
        labels=['Small', 'Medium', 'Large']
    )
    
    print(f"✅ Feature engineering completed. New features added:")
    new_features = ['sepal_ratio', 'petal_ratio', 'sepal_area', 'petal_area', 
                    'total_size', 'sepal_size_category', 'petal_size_category']
    for feature in new_features:
        print(f"   - {feature}")
    
    return df_processed


def main():
    args = parse_args()
    
    print("Starting feature engineering...")
    print(f"Input training data path: {args.input_train_data}")
    print(f"Input test data path: {args.input_test_data}")
    
    # Load the datasets
    train_files = [f for f in os.listdir(args.input_train_data) if f.endswith('.csv')]
    test_files = [f for f in os.listdir(args.input_test_data) if f.endswith('.csv')]
    
    print(f"Training files found: {train_files}")
    print(f"Test files found: {test_files}")
    
    # Process training data
    train_dfs = []
    for file in train_files:
        df = pd.read_csv(Path(args.input_train_data) / file)
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # Process test data
    test_dfs = []
    for file in test_files:
        df = pd.read_csv(Path(args.input_test_data) / file)
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Apply feature engineering
    train_processed = engineer_features(train_df)
    test_processed = engineer_features(test_df)
    
    # Create output directories
    os.makedirs(args.output_train_data, exist_ok=True)
    os.makedirs(args.output_test_data, exist_ok=True)
    
    # Save processed data
    train_path = Path(args.output_train_data) / "processed_train.csv"
    test_path = Path(args.output_test_data) / "processed_test.csv"
    
    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
    
    print(f"✅ Processed training data saved to: {train_path}")
    print(f"✅ Processed test data saved to: {test_path}")
    print("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()