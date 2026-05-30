"""
Download and load real-world time series datasets.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional
from urllib.request import urlretrieve


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_electricity_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download and load the UCI Electricity Load Dataset.
    This dataset contains electricity consumption of 370 clients.
    
    Args:
        data_dir: Directory to save/load data
        
    Returns:
        DataFrame with electricity consumption data
    """
    ensure_dir(data_dir)
    data_path = os.path.join(data_dir, "electricity.csv")
    
    if not os.path.exists(data_path):
        print("Downloading Electricity Load Dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
        zip_path = os.path.join(data_dir, "LD2011_2014.txt.zip")
        urlretrieve(url, zip_path)
        
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        txt_path = os.path.join(data_dir, "LD2011_2014.txt")
        df = pd.read_csv(txt_path, sep=';', index_col=0, parse_dates=True)
        df = df.replace(',', '.', regex=True).astype(float)
        df.to_csv(data_path)
        os.remove(zip_path)
        os.remove(txt_path)
        print(f"Data saved to {data_path}")
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    return df


def download_air_quality_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download and load Beijing Air Quality Dataset.
    
    Args:
        data_dir: Directory to save/load data
        
    Returns:
        DataFrame with air quality data
    """
    ensure_dir(data_dir)
    data_path = os.path.join(data_dir, "air_quality.csv")
    
    if not os.path.exists(data_path):
        print("Downloading Air Quality Dataset...")
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.csv",
        ]
        dfs = []
        for i, url in enumerate(urls):
            csv_path = os.path.join(data_dir, f"air_quality_{i}.csv")
            if not os.path.exists(csv_path):
                urlretrieve(url, csv_path)
            df_part = pd.read_csv(csv_path)
            dfs.append(df_part)
        
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")
    else:
        df = pd.read_csv(data_path)
    
    return df


def generate_stock_like_data(num_samples: int = 5000, num_features: int = 5) -> np.ndarray:
    """
    Generate synthetic stock-like time series data with realistic characteristics.
    
    Args:
        num_samples: Number of time steps
        num_features: Number of features
        
    Returns:
        Synthetic time series data of shape (num_samples, num_features)
    """
    np.random.seed(42)
    data = np.zeros((num_samples, num_features))
    
    for i in range(num_features):
        # Random walk with trend
        trend = np.linspace(0, np.random.randn() * 5, num_samples)
        noise = np.random.normal(0, 0.1, num_samples)
        data[:, i] = np.cumsum(noise) + trend
        
        # Add seasonality
        seasonality = np.sin(np.linspace(0, 10 * np.pi, num_samples)) * 0.5
        data[:, i] += seasonality
    
    return data


def prepare_univariate_data(
    df: pd.DataFrame, 
    column: str, 
    seq_len: int = 32,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare univariate time series data for training.
    
    Args:
        df: Input DataFrame
        column: Column name to use
        seq_len: Sequence length
        horizon: Prediction horizon
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        (train_data, val_data, test_data) as numpy arrays
    """
    series = df[column].values.astype(np.float32)
    
    # Handle NaNs
    series = np.nan_to_num(series, nan=np.nanmean(series))
    
    # Normalize
    mean = np.mean(series)
    std = np.std(series) + 1e-8
    series = (series - mean) / std
    
    split_train = int(len(series) * train_ratio)
    split_val = int(len(series) * (train_ratio + val_ratio))
    
    train_data = series[:split_train]
    val_data = series[split_train:split_val]
    test_data = series[split_val:]
    
    return train_data, val_data, test_data


def create_real_dataloader(
    data: np.ndarray,
    seq_len: int = 32,
    horizon: int = 1,
    batch_size: int = 32,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader from numpy array.
    
    Args:
        data: Input time series data
        seq_len: Sequence length
        horizon: Prediction horizon
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        PyTorch DataLoader
    """
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len, np.newaxis])
        y.append(data[i + seq_len:i + seq_len + horizon, np.newaxis])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
