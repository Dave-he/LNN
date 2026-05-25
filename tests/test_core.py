import numpy as np
import torch

from lnn.core.cfc import CfCCell, CfCNetwork
from lnn.core.liquid_neuron import LiquidLayer, LiquidNeuron, LiquidNN
from lnn.core.ltc import LTCCell, LTCNetwork
from lnn.data.timeseries import TimeSeriesDataset, generate_mackey_glass, generate_sine_data
from lnn.utils.metrics import compute_metrics


class TestLiquidNeuron:
    def test_output_shape(self):
        model = LiquidNeuron(input_size=4, hidden_size=8)
        x = torch.randn(2, 10, 4)
        out, h = model(x)
        assert out.shape == (2, 10, 8)
        assert h.shape == (2, 8)

    def test_custom_h0(self):
        model = LiquidNeuron(input_size=4, hidden_size=8)
        x = torch.randn(2, 10, 4)
        h0 = torch.zeros(2, 8)
        out, h = model(x, h0=h0)
        assert out.shape == (2, 10, 8)


class TestLiquidLayer:
    def test_output_shape(self):
        model = LiquidLayer(input_size=4, hidden_size=8, ode_method="euler")
        x = torch.randn(2, 5, 4)
        out, h = model(x)
        assert out.shape == (2, 5, 8)
        assert h.shape == (2, 8)


class TestLiquidNN:
    def test_output_shape(self):
        model = LiquidNN(input_size=4, hidden_size=8, output_size=2, num_layers=1, ode_method="euler")
        x = torch.randn(2, 5, 4)
        out = model(x)
        assert out.shape == (2, 5, 2)


class TestLTCCell:
    def test_output_shape(self):
        cell = LTCCell(input_size=4, hidden_size=8, ode_method="euler")
        x_t = torch.randn(2, 4)
        h = torch.zeros(2, 8)
        h_new = cell(x_t, h)
        assert h_new.shape == (2, 8)


class TestLTCNetwork:
    def test_output_shape(self):
        model = LTCNetwork(input_size=4, hidden_size=8, output_size=2, ode_method="euler")
        x = torch.randn(2, 5, 4)
        out = model(x)
        assert out.shape == (2, 5, 2)


class TestCfCCell:
    def test_output_shape(self):
        cell = CfCCell(input_size=4, hidden_size=8)
        x_t = torch.randn(2, 4)
        h = torch.zeros(2, 8)
        h_new = cell(x_t, h)
        assert h_new.shape == (2, 8)

    def test_no_ode_solver(self):
        cell = CfCCell(input_size=4, hidden_size=8)
        x_t = torch.randn(2, 4)
        h = torch.zeros(2, 8)
        h_new = cell(x_t, h)
        assert not torch.isnan(h_new).any()


class TestCfCNetwork:
    def test_output_shape_sequence(self):
        model = CfCNetwork(input_size=4, hidden_size=8, output_size=2, return_sequences=True)
        x = torch.randn(2, 5, 4)
        out = model(x)
        assert out.shape == (2, 5, 2)

    def test_output_shape_last(self):
        model = CfCNetwork(input_size=4, hidden_size=8, output_size=2, return_sequences=False)
        x = torch.randn(2, 5, 4)
        out = model(x)
        assert out.shape == (2, 2)


class TestDataGeneration:
    def test_sine_data(self):
        data = generate_sine_data(num_samples=100)
        assert data.shape == (100,)
        assert data.dtype.name == "float32"

    def test_mackey_glass(self):
        data = generate_mackey_glass(num_samples=100)
        assert data.shape == (100,)
        assert not np.isnan(data).any()


class TestTimeSeriesDataset:
    def test_dataset_length(self):
        data = generate_sine_data(num_samples=200)
        ds = TimeSeriesDataset(data, seq_len=32, horizon=1)
        assert len(ds) > 0

    def test_dataset_shapes(self):
        data = generate_sine_data(num_samples=200)
        ds = TimeSeriesDataset(data, seq_len=32, horizon=1)
        x, y = ds[0]
        assert x.shape == (32, 1)
        assert y.shape == (1,)


class TestMetrics:
    def test_compute_metrics(self):
        import numpy as np
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1])
        m = compute_metrics(y_true, y_pred)
        assert "mse" in m
        assert "rmse" in m
        assert "mae" in m
        assert m["mse"] > 0
        assert m["rmse"] > 0
