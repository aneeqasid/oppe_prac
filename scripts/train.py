import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import argparse
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def impute_missing(df, feature_cols, target_col='species'):
    """Impute missing feature values using mean of last 10 same-species samples"""
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            for species in df[target_col].unique():
                mask = df[target_col] == species
                species_rows = df.loc[mask & df[col].notna()]
                if len(species_rows) >= 10:
                    mean_val = species_rows[col].tail(10).mean()
                else:
                    mean_val = species_rows[col].mean() if len(species_rows) > 0 else df[col].mean()
                df.loc[mask & df[col].isna(), col] = mean_val
    return df

def load_data(versions):
    dfs = []
    for v in versions:
        path = f"{DATA_DIR}/iris_v{v}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No data files found")
    return pd.concat(dfs, ignore_index=True)

def train_iteration(version_list):
    print(f"\n=== Loading data for versions: {version_list} ===")
    df = load_data(version_list)
    
    # --- FIX: Drop rows where species is missing before encoding ---
    df = df.dropna(subset=['species'])
    
    # Encode species
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['target'] = df['species'].map(species_map)
    
    # Drop any remaining NaN in target (just in case)
    df = df.dropna(subset=['target'])
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Impute missing feature values
    df = impute_missing(df, feature_cols)
    
    # Split
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_model = None
    best_acc = 0
    best_params = {}
    
    # Hyperparameter tuning
    for n_est in [50, 100]:
        for depth in [None, 10]:
            with mlflow.start_run(nested=True):
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", depth)
                mlflow.log_param("data_versions", str(version_list))
                
                model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                mlflow.log_metric("accuracy", acc)
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {'n_estimators': n_est, 'max_depth': depth}
    
    # Log best model
    with mlflow.start_run(run_name=f"Best_v{''.join(map(str, version_list))}"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_acc)
        mlflow.sklearn.log_model(best_model, "model")
        
        model_path = f"{MODEL_DIR}/model_v{''.join(map(str, version_list))}.pkl"
        joblib.dump(best_model, model_path)
        
        print(f"✅ Best accuracy: {best_acc:.3f} for versions {version_list}")
        print(f"   Params: {best_params}")
    
    return best_model, best_acc

if __name__ == "__main__":
    # Set experiment
    mlflow.set_experiment("Iris_Classifier_OPPE")
    
    # Train Iteration 1: Only v0
    print("\n=== Training on versions: [0] ===")
    train_iteration([0])
    
    # Train Iteration 2: Merge v0 + v1
    print("\n=== Training on versions: [0, 1] ===")
    train_iteration([0, 1])
