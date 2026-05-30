#!/usr/bin/env python3
"""
快速测试和验证所有液态神经网络模型的实现
"""

import sys
import os
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np

from lnn.core.ltc import LTCNetwork
from lnn.core.cfc import CfCNetwork
from lnn.core.variants import (
    StrictCfCCell, StrictCfCNetwork,
    HybridCfCCell, HybridCfCNetwork,
    CTLTCCell, CTLTCNetwork,
    LiquidS4Cell, LiquidS4Network,
    LRCCell, LRCNetwork,
    CfCDTCell, CfCDTNetwork,
    EulerLTCDTCell, EulerLTCDTNetwork
)

def test_forward_pass(name, model_class, use_dt=False):
    """测试模型的前向传播是否正常工作"""
    print(f"\n{'='*60}")
    print(f"测试模型: {name}")
    print(f"{'='*60}")
    
    try:
        input_size = 1
        hidden_size = 8
        output_size = 1
        num_layers = 1
        
        # 创建模型
        model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers
        )
        
        # 生成测试输入
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, input_size)
        
        start_time = time.time()
        
        # 前向传播
        if use_dt:
            dt = torch.rand(batch_size, seq_len, 1) + 0.5
            output = model(x, dt=dt)
        else:
            output = model(x)
        
        forward_time = time.time() - start_time
        
        # 检查输出形状
        print(f"✓ 模型创建成功")
        print(f"✓ 前向传播成功, 耗时: {forward_time:.6f}s")
        
        if hasattr(model, 'return_sequences') and model.return_sequences:
            expected_shape = (batch_size, seq_len, output_size)
        else:
            expected_shape = (batch_size, output_size)
        
        if output.shape == expected_shape:
            print(f"✓ 输出形状正确: {output.shape}")
        else:
            print(f"✗ 输出形状不正确! 期望: {expected_shape}, 实际: {output.shape}")
            return False
        
        # 检查输出值是否合理
        if torch.isnan(output).any():
            print("✗ 输出包含 NaN!")
            return False
        
        if torch.isinf(output).any():
            print("✗ 输出包含 inf!")
            return False
        
        print("✓ 输出值合理")
        
        # 测试反向传播
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        target = torch.randn_like(output)
        loss_fn = nn.MSELoss()
        
        optimizer.zero_grad()
        
        if use_dt:
            pred = model(x, dt=dt)
        else:
            pred = model(x)
        
        if pred.dim() > target.dim():
            target = target.unsqueeze(1).repeat(1, pred.shape[1], 1)
        
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        
        print("✓ 反向传播和优化成功")
        
        # 统计参数数量
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ 模型参数数量: {param_count:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_models():
    """测试所有模型"""
    print("="*60)
    print("液态神经网络模型测试")
    print("="*60)
    
    results = []
    
    # 测试标准模型
    results.append(("LTC", LTCNetwork, False))
    results.append(("CfC", CfCNetwork, False))
    
    # 测试变体模型
    results.append(("StrictCfC", StrictCfCNetwork, False))
    results.append(("HybridCfC", HybridCfCNetwork, False))
    results.append(("CTLTC", CTLTCNetwork, False))
    results.append(("LiquidS4", LiquidS4Network, False))
    results.append(("LRC", LRCNetwork, False))
    
    # 测试支持 dt 的模型
    results.append(("CfC-DT", CfCDTNetwork, True))
    results.append(("Euler-LTC-DT", EulerLTCDTNetwork, True))
    
    # 运行所有测试
    all_passed = True
    test_results = []
    
    for name, model_class, use_dt in results:
        passed = test_forward_pass(name, model_class, use_dt=use_dt)
        test_results.append((name, passed))
        if not passed:
            all_passed = False
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, passed in test_results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有模型测试通过!")
    else:
        print("✗ 部分模型测试失败!")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = test_all_models()
    sys.exit(0 if success else 1)
