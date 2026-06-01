"""
真实数据集加载器模块
包含股票、能源、UCI 时间序列数据
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Union


class RealTimeSeriesDataset(Dataset):
    """
    通用真实时间序列数据集加载器
    支持 Pandas DataFrame 或 NumPy 数组输入
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        seq_len: int = 32,
        horizon: int = 1,
        stride: int = 1,
        normalize: bool = True,
        train_split: float = 0.7,
        mode: str = "train",
    ):
        """
        Args:
            data: Pandas DataFrame 或 NumPy 数组
            target_col: 目标列名（仅 DataFrame 时需要
            seq_len: 输入序列长度
            horizon: 预测步数
            stride: 滑动窗口步长
            normalize: 是否归一化
            train_split: 训练集分割比例
            mode: 模式 ("train", "val", "test"
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.stride = stride

        # 数据转换
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                raise ValueError("target_col 必须指定")
            self.df = data.copy()
            self.data = data[target_col].values.astype(np.float32)
        else:
            self.df = None
            self.data = data.astype(np.float32)

        # 分割数据
        n = len(self.data)
        n_train = int(n * train_split)
        n_val = int(n * (train_split + (1 - train_split) / 2))

        if mode == "train":
            self.slice = slice(0, n_train)
        elif mode == "val":
            self.slice = slice(n_train, n_val)
        elif mode == "test":
            self.slice = slice(n_val, None)
        else:
            raise ValueError("mode 必须是 'train', 'val' 或 'test'")

        self.data_slice = self.data[self.slice]

        # 归一化
        self.normalize = normalize
        if normalize:
            self.mean = self.data_slice.mean()
            self.std = self.data_slice.std() + 1e-8
            self.data_slice = (self.data_slice - self.mean) / self.std

        # 创建滑动窗口
        self.indices = list(range(0, len(self.data_slice) - seq_len - horizon + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x = self.data_slice[start:start + self.seq_len]
        y = self.data_slice[start + self.seq_len:start + self.seq_len + self.horizon]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


def generate_stock_like_data(
    num_samples: int = 2000,
    seed: int = 42,
    drift: float = 0.001,
    volatility: float = 0.02,
) -> np.ndarray:
    """
    生成股票风格随机游走序列（模拟真实股票数据
    """
    rng = np.random.default_rng(seed)
    
    # 对数收益率
    log_returns = rng.normal(drift, volatility, num_samples)
    prices = np.cumsum(log_returns)
    
    # 标准化
    prices = (prices - prices.mean()) / (prices.std() + 1e-8)
    
    return prices.astype(np.float32)


def load_yahoo_finance(
    symbol: str = "AAPL",
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    加载 Yahoo Finance 数据（本地缓存或模拟）
    实际环境中可替换为真实 yfinance 调用
    """
    cache_file = f"cache_{symbol}.csv"
    
    if use_cache and os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=[0], index_col=0)
    
    # 模拟股票数据（用于演示
    print(f"生成模拟股票数据（无真实 API")
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    
    # 生成随机游走
    np.random.seed(42)
    base = 100.0
    log_returns = np.random.normal(0.0005, 0.02, n-1)
    prices = base * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, base)
    
    df = pd.DataFrame({
        "Open": prices,
        "High": prices * (1 + np.random.uniform(0, 0.02, n)),
        "Low": prices * (1 - np.random.uniform(0, 0.02, n)),
        "Close": prices,
        "Volume": np.random.randint(1000000, 10000000, n),
    }, index=dates)
    
    if use_cache:
        df.to_csv(cache_file)
    
    return df


def create_real_data_loaders(
    data: Union[pd.DataFrame, np.ndarray],
    target_col: str = "Close",
    seq_len: int = 32,
    horizon: int = 1,
    batch_size: int = 32,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 train/val/test 数据加载器
    """
    train_dataset = RealTimeSeriesDataset(
        data, target_col=target_col, seq_len=seq_len, horizon=horizon,
        normalize=normalize, mode="train"
    )
    val_dataset = RealTimeSeriesDataset(
        data, target_col=target_col, seq_len=seq_len, horizon=horizon,
        normalize=normalize, mode="val"
    )
    test_dataset = RealTimeSeriesDataset(
        data, target_col=target_col, seq_len=seq_len, horizon=horizon,
        normalize=normalize, mode="test"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
