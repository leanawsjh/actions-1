# src/train.py

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(test_size: float, random_state:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def main():
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    print(f" Target samples - Train: {y_train.shape[0]}, Test: {y_test.shape[0]}")

if __name__ == "__main__":
    main()