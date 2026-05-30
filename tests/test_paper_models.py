import torch
import numpy as np
from lnn.core.paper_models import (
    LSTMModel,
    StrictCfCModel,
    LTCModel,
    HybridCfCModel,
    CTLTCModel,
    MSCfCModel,
    VolatilityWeightedMSELoss
)


def test_lstm_model():
    batch_size = 4
    seq_len = 30
    input_size = 30
    hidden_size = 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    model = LSTMModel(input_size, hidden_size)
    
    pred = model(x)
    assert pred.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {pred.shape}"
    
    # Test gradient flow
    loss = pred.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"


def test_strict_cfc_model():
    batch_size = 4
    seq_len = 30
    input_size = 30
    hidden_size = 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    model = StrictCfCModel(input_size, hidden_size)
    
    pred = model(x)
    assert pred.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {pred.shape}"
    
    # Test gradient flow
    loss = pred.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"


def test_ltc_model():
    batch_size = 4
    seq_len = 30
    input_size = 30
    hidden_size = 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    model = LTCModel(input_size, hidden_size, l_ode=6)
    
    pred = model(x)
    assert pred.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {pred.shape}"
    
    # Test gradients and check parameters
    loss = pred.sum()
    loss.backward()
    assert model.theta_tau.grad is not None, "theta_tau did not receive gradients"
    assert model.A.grad is not None, "attractor A did not receive gradients"


def test_hybrid_cfc_model():
    batch_size = 4
    seq_len = 30
    input_size = 30
    hidden_size = 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    model = HybridCfCModel(input_size, hidden_size)
    
    pred = model(x)
    assert pred.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {pred.shape}"
    
    loss = pred.sum()
    loss.backward()
    assert model.theta_tau.grad is not None, "theta_tau did not receive gradients"


def test_ct_ltc_model():
    batch_size = 4
    seq_len = 30
    input_size = 29  # CT-LTC excludes lagged return
    hidden_size = 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    # Generate random daily gaps (e.g. 1.0 or 3.0 for weekend gap)
    dt = torch.clamp(torch.randint(1, 4, (batch_size, seq_len)).float(), min=1.0)
    
    model = CTLTCModel(input_size, hidden_size, l_ode=6)
    pred = model(x, dt)
    assert pred.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {pred.shape}"
    
    loss = pred.sum()
    loss.backward()
    assert model.theta_tau.grad is not None, "theta_tau did not receive gradients"


def test_ms_cfc_model():
    batch_size = 4
    seq_len = 30
    input_size = 30
    hidden_size = 12
    
    x = torch.randn(batch_size, seq_len, input_size)
    model = MSCfCModel(input_size, hidden_size)
    
    pred = model(x)
    assert pred.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {pred.shape}"
    
    loss = pred.sum()
    loss.backward()
    assert model.theta_tau.grad is not None, "theta_tau did not receive gradients"


def test_volatility_weighted_loss():
    batch_size = 4
    pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    target = torch.tensor([[1.1], [1.9], [3.2], [3.8]])
    rolling_vol = torch.tensor([[0.5], [1.0], [2.5], [0.2]])  # Different volatilities
    
    loss_fn = VolatilityWeightedMSELoss(gamma=2.0)
    loss = loss_fn(pred, target, rolling_vol)
    
    # Manually compute
    weights = 1.0 + 2.0 * np.array([0.5, 1.0, 2.5, 0.2])
    errors = (np.array([1.0, 2.0, 3.0, 4.0]) - np.array([1.1, 1.9, 3.2, 3.8])) ** 2
    expected_loss = np.mean(errors * weights)
    
    assert np.allclose(loss.item(), expected_loss), f"Expected {expected_loss}, got {loss.item()}"


if __name__ == "__main__":
    print("Running unit tests for paper models...")
    try:
        print("1. Testing LSTMModel...")
        test_lstm_model()
        print("   OK")
        
        print("2. Testing StrictCfCModel...")
        test_strict_cfc_model()
        print("   OK")
        
        print("3. Testing LTCModel...")
        test_ltc_model()
        print("   OK")
        
        print("4. Testing HybridCfCModel...")
        test_hybrid_cfc_model()
        print("   OK")
        
        print("5. Testing CTLTCModel...")
        test_ct_ltc_model()
        print("   OK")
        
        print("6. Testing MSCfCModel (Optimization Strategy 1)...")
        test_ms_cfc_model()
        print("   OK")
        
        print("7. Testing VolatilityWeightedMSELoss (Optimization Strategy 2)...")
        test_volatility_weighted_loss()
        print("   OK")
        
        print("\nAll unit tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
