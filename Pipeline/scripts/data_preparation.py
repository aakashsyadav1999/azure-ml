"""
Data preparation component for Iris classification
Generates and preprocesses the Iris dataset for training
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os


def parse_args():
    parser = argparse.ArgumentParser("data-prep")
    parser.add_argument("--output_train_data", type=str, help="Path to save training data")
    parser.add_argument("--output_test_data", type=str, help="Path to save test data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def generate_iris_data(test_size=0.2, random_state=42):
    """Generate and split Iris dataset"""
    print("Loading Iris dataset...")
    iris = load_iris()
    
    # Create DataFrame with feature names
    feature_names = iris.feature_names
    df = pd.DataFrame(iris.data, columns=feature_names)
    df['species'] = iris.target
    
    # Map target numbers to species names
    target_names = iris.target_names
    df['species_name'] = df['species'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {feature_names}")
    print(f"Target distribution:\n{df['species_name'].value_counts()}")
    
    # Split the data
    X = df.drop(['species'], axis=1)
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Combine features and target back
    train_df = X_train.copy()
    train_df['species'] = y_train
    
    test_df = X_test.copy()
    test_df['species'] = y_test
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, test_df


def main():
    args = parse_args()
    
    print("Starting data preparation...")
    print(f"Output training data path: {args.output_train_data}")
    print(f"Output test data path: {args.output_test_data}")
    
    # Generate the dataset
    train_df, test_df = generate_iris_data(
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Create output directories
    os.makedirs(args.output_train_data, exist_ok=True)
    os.makedirs(args.output_test_data, exist_ok=True)
    
    # Save the datasets
    train_path = Path(args.output_train_data) / "train.csv"
    test_path = Path(args.output_test_data) / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Training data saved to: {train_path}")
    print(f"✅ Test data saved to: {test_path}")
    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()