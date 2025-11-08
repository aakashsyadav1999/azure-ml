"""
Data generation script - fetches Iris dataset
"""
import pandas as pd
import os
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def generate_iris_data(output_dir="data", test_size=0.2, random_state=42):
    """Load Iris dataset and save as CSV files"""
    iris = load_iris()
    
    # Create DataFrame
    feature_names = [c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names]
    df = pd.DataFrame(iris.data, columns=feature_names)
    df["target"] = iris.target
    df["target_name"] = [iris.target_names[int(t)] for t in iris.target]
    
    # Split and save
    os.makedirs(output_dir, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["target"])
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data: {train_path} ({len(train_df)} rows)")
    print(f"Test data: {test_path} ({len(test_df)} rows)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    
    args = parser.parse_args()
    generate_iris_data(args.output_dir, args.test_size, args.random_state)

if __name__ == "__main__":
    main()
