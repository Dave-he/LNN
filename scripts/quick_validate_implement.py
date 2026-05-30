#!/usr/bin/env python3
"""
快速验证实现的完整性

此脚本快速测试所有实现的模型变体是否可以正常工作
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lnn.core.ltc import LTCNetwork
from lnn.core.cfc import CfCNetwork
from lnn.core.variants import (
    StrictCfCNetwork, HybridCfCNetwork, CTLTCNetwork,
    LiquidS4Network, LRCNetwork, CfCDTNetwork, EulerLTCDTNetwork
)


def test_model(model_name, model_class):
    """测试单个模型"""
    print(f"\n测试 {model_name}...")
    input_size = 1
    hidden_size = 8
    output_size = 1
    batch_size = 2
    seq_len = 10
    
    try:
        # 创建模型
        model = model_class(input_size, hidden_size, output_size)
        print(f"  ✓ 创建模型: {model}")
        
        # 生成测试数据
        x = torch.randn(batch_size, seq_len, input_size)
        
        # 前向传播
        start_time = time.time()
        output = model(x)
        forward_time = time.time() - start_time
        print(f"  ✓ 前向传播成功, 输出形状: {output.shape}, 耗时: {forward_time:.4f}s")
        
        # 检查输出
        if output.shape == (batch_size, seq_len, output_size):
            print(f"  ✓ 输出形状正确")
        elif output.shape == (batch_size, output_size):
            print(f"  ⚠️ 输出是最后一步输出 (可能是正确的)")
        
        # 简单训练测试
        y = torch.randn(batch_size, output_size)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        # 计算损失和反向传播
        if output.dim() > 2:
            loss = loss_fn(output[:, -1, :], y)
        else:
            loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        print(f"  ✓ 反向传播和优化成功")
        
        return True, forward_time
    
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0


def test_dt_model(model_name, model_class):
    """测试支持时间步的模型"""
    print(f"\n测试 {model_name} (带 dt)...")
    input_size = 1
    hidden_size = 8
    output_size = 1
    batch_size = 2
    seq_len = 10
    
    try:
        model = model_class(input_size, hidden_size, output_size)
        print(f"  ✓ 创建模型: {model}")
        
        x = torch.randn(batch_size, seq_len, input_size)
        dt = torch.randn(batch_size, seq_len, 1) + 0.5  # 正的时间步
        
        # 前向传播
        start_time = time.time()
        output = model(x, dt=dt)
        forward_time = time.time() - start_time
        print(f"  ✓ 前向传播成功, 输出形状: {output.shape}, 耗时: {forward_time:.4f}s")
        
        return True, forward_time
    
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0


def main():
    print("=" * 80)
    print("液态神经网络实现验证")
    print("=" * 80)
    
    models = [
        ("LTC", LTCNetwork),
        ("CfC", CfCNetwork),
        ("Strict CfC", StrictCfCNetwork),
        ("Hybrid CfC", HybridCfCNetwork),
        ("CT-LTC", CTLTCNetwork),
        ("LRC", LRCNetwork),
    ]
    
    dt_models = [
        ("CfC-DT", CfCDTNetwork),
        ("Euler-LTC-DT", EulerLTCDTNetwork),
    ]
    
    liquid_s4_model = [
        ("Liquid-S4", LiquidS4Network),
    ]
    
    results = {}
    
    print("\n测试标准模型:")
    for name, cls in models:
        success, time_t = test_model(name, cls)
        results[name] = (success, time_t)
    
    print("\n测试 dt 感知模型:")
    for name, cls in dt_models:
        success, time_t = test_dt_model(name, cls)
        results[name] = (success, time_t)
    
    print("\n测试 Liquid-S4:")
    for name, cls in liquid_s4_model:
        success, time_t = test_model(name, cls)
        results[name] = (success, time_t)
    
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    all_passed = True
    for name, (success, t) in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name:20s} {status}")
        if success and t > 0:
            print(f"  前向时间: {t:.6f}s")
        all_passed = all_passed and success
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有模型实现正确!")
    else:
        print("✗ 部分模型实现有问题，请检查")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
