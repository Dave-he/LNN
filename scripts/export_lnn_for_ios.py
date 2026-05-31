
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lnn.core.cfc import CfCNetwork
from lnn.data.timeseries import generate_sine_data, create_dataloader


def train_and_export_cfc_model():
    # 设置设备
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 生成训练数据 - 正弦波
    print("Generating training data...")
    data = generate_sine_data(num_samples=2000, freq=0.1)
    train_data = data[:1600]
    test_data = data[1600:]
    
    seq_len = 16
    horizon = 1
    
    train_loader = create_dataloader(train_data, seq_len=seq_len, horizon=horizon, batch_size=32)
    test_loader = create_dataloader(test_data, seq_len=seq_len, horizon=horizon, batch_size=32)
    
    # 创建 CfC 模型
    print("Creating CfC model...")
    model = CfCNetwork(
        input_size=1,
        hidden_size=8,
        output_size=1,
        num_layers=1,
        return_sequences=False
    )
    model.to(device)
    
    # 训练模型
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    
    print(f"Training model for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    # 在测试集上评估
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            test_loss += criterion(outputs, y).item()
    
    print(f"Test Loss: {test_loss/len(test_loader):.6f}")
    
    # 保存完整模型（用于转换）
    model_path = "/Users/hyx/workspace/LNN/ios/LNNDemo/LNNDemo/Models/cfc_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 创建一个简化的可导出版本（兼容 TorchScript）
    class ExportableCfC(nn.Module):
        def __init__(self, hidden_size=8):
            super().__init__()
            self.hidden_size = hidden_size
            
            self.f_gate = nn.Sequential(
                nn.Linear(1 + hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.g_branch = nn.Sequential(
                nn.Linear(1 + hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.h_branch = nn.Sequential(
                nn.Linear(1 + hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.time_scale = nn.Parameter(torch.ones(hidden_size))
            self.output_proj = nn.Linear(hidden_size, 1)
        
        def forward(self, x_seq, h0=None):
            batch_size = x_seq.shape[0]
            seq_len = x_seq.shape[1]
            
            if h0 is None:
                h0 = torch.zeros(batch_size, self.hidden_size, device=x_seq.device)
            
            h = h0
            for t in range(seq_len):
                x_t = x_seq[:, t, :]
                combined = torch.cat([x_t, h], dim=-1)
                f = self.f_gate(combined)
                g = self.g_branch(combined)
                h_out = self.h_branch(combined)
                decay = torch.sigmoid(-f * self.time_scale * 1.0)
                h = decay * g + (1.0 - decay) * h_out
            
            return self.output_proj(h)
    
    # 复制权重到可导出模型
    exportable_model = ExportableCfC(hidden_size=8)
    with torch.no_grad():
        exportable_model.f_gate[0].weight.copy_(model.cells[0].f_gate[0].weight)
        exportable_model.f_gate[0].bias.copy_(model.cells[0].f_gate[0].bias)
        exportable_model.g_branch[0].weight.copy_(model.cells[0].g_branch[0].weight)
        exportable_model.g_branch[0].bias.copy_(model.cells[0].g_branch[0].bias)
        exportable_model.h_branch[0].weight.copy_(model.cells[0].h_branch[0].weight)
        exportable_model.h_branch[0].bias.copy_(model.cells[0].h_branch[0].bias)
        exportable_model.time_scale.copy_(model.cells[0].time_scale)
        exportable_model.output_proj.weight.copy_(model.output_proj.weight)
        exportable_model.output_proj.bias.copy_(model.output_proj.bias)
    
    # 导出 TorchScript
    exportable_model.eval()
    example_input = torch.randn(1, seq_len, 1)
    traced_model = torch.jit.trace(exportable_model, example_input)
    traced_path = "/Users/hyx/workspace/LNN/ios/LNNDemo/LNNDemo/Models/cfc_model_traced.pt"
    traced_model.save(traced_path)
    print(f"Traced model saved to {traced_path}")
    
    # 保存一些示例数据用于测试
    example_data = {
        "seq_len": seq_len,
        "test_input": test_data[:seq_len].tolist(),
        "test_output": test_data[seq_len:seq_len+horizon].tolist()
    }
    import json
    with open("/Users/hyx/workspace/LNN/ios/LNNDemo/LNNDemo/Models/example_data.json", "w") as f:
        json.dump(example_data, f)
    
    print("Export complete!")
    return exportable_model


if __name__ == "__main__":
    train_and_export_cfc_model()
