import numpy as np
import torch

from lnn.core.cfc import CfCCell, CfCNetwork
from lnn.core.control import LNNImitationPolicy
from lnn.core.graph import GraphLNNPredictor, GraphSnapshotEncoder
from lnn.core.liquid_neuron import LiquidLayer, LiquidNeuron, LiquidNN
from lnn.core.long_sequence import LiquidS4Block, LiquidTADHead, LongSequenceLiquidClassifier
from lnn.core.ltc import LTCCell, LTCNetwork
from lnn.core.mdn import MDNHead, mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal import MultimodalFusionLNN
from lnn.core.physics import PhysicsInformedLNN, damped_oscillator_residual, physics_informed_loss
from lnn.core.trainer import Trainer
from lnn.data.graph_timeseries import SyntheticGraphTimeSeriesDataset, create_graph_dataloaders
from lnn.data.long_sequence import SyntheticLongSequenceDataset, create_long_sequence_dataloaders
from lnn.data.multimodal import SyntheticMultimodalDataset, create_multimodal_dataloaders
from lnn.data.physics import DampedOscillatorDataset, create_physics_dataloaders
from lnn.data.robotics import SyntheticImitationDataset, create_imitation_dataloaders
from lnn.data.timeseries import TimeSeriesDataset, create_dataloader, generate_mackey_glass, generate_sine_data
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

    def test_tensor_dt(self):
        cell = CfCCell(input_size=4, hidden_size=8)
        x_t = torch.randn(2, 4)
        h = torch.zeros(2, 8)
        dt = torch.tensor([[0.2], [1.5]])
        h_new = cell(x_t, h, dt=dt)
        assert h_new.shape == (2, 8)


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

    def test_dt_and_mask_forward(self):
        model = CfCNetwork(input_size=4, hidden_size=8, output_size=2, return_sequences=True)
        x = torch.randn(2, 5, 4)
        x[:, 2, :] = float("nan")
        dt = torch.rand(2, 5, 1) + 0.1
        mask = torch.ones(2, 5, 4)
        mask[:, 2, :] = 0.0
        out = model(x, dt=dt, mask=mask)
        assert out.shape == (2, 5, 2)
        assert torch.isfinite(out).all()


class TestMultimodalFusionLNN:
    def test_forward_shape(self):
        dataset = SyntheticMultimodalDataset(num_samples=8, seq_len=6, sensor_dim=3, image_size=8, text_len=5)
        batch, _ = next(iter(create_multimodal_dataloaders(dataset, batch_size=4)[0]))
        model = MultimodalFusionLNN(
            sensor_dim=3,
            image_channels=1,
            vocab_size=48,
            num_classes=3,
            fusion_size=8,
            hidden_size=12,
        )
        logits = model(batch)
        assert logits.shape == (4, 3)

    def test_training_step(self):
        dataset = SyntheticMultimodalDataset(num_samples=12, seq_len=5, sensor_dim=2, image_size=8, text_len=4)
        batch, labels = next(iter(create_multimodal_dataloaders(dataset, batch_size=6)[0]))
        model = MultimodalFusionLNN(
            sensor_dim=2,
            image_channels=1,
            vocab_size=48,
            num_classes=3,
            fusion_size=8,
            hidden_size=10,
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    def test_ltc_forward_shape(self):
        dataset = SyntheticMultimodalDataset(num_samples=8, seq_len=3, sensor_dim=2, image_size=8, text_len=4)
        batch, _ = next(iter(create_multimodal_dataloaders(dataset, batch_size=4)[0]))
        model = MultimodalFusionLNN(
            sensor_dim=2,
            image_channels=1,
            vocab_size=48,
            num_classes=3,
            fusion_size=6,
            hidden_size=8,
            recurrent_type="ltc",
        )
        logits = model(batch)
        assert logits.shape == (4, 3)


class TestMDNHead:
    def test_shapes_and_loss(self):
        head = MDNHead(input_size=6, output_size=2, num_mixtures=3)
        features = torch.randn(5, 6)
        target = torch.randn(5, 2)
        params = head(features)
        assert params["logits"].shape == (5, 3)
        assert params["loc"].shape == (5, 3, 2)
        assert params["log_scale"].shape == (5, 3, 2)
        loss = mdn_negative_log_likelihood(params, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert mdn_mean(params).shape == (5, 2)


class TestLNNImitationPolicy:
    def test_mse_policy_shape(self):
        model = LNNImitationPolicy(state_dim=6, action_dim=2, hidden_size=8, head_type="mse")
        states = torch.randn(4, 7, 6)
        actions = model(states)
        assert actions.shape == (4, 2)

    def test_mdn_policy_shape_with_metadata(self):
        model = LNNImitationPolicy(state_dim=6, action_dim=2, hidden_size=8, head_type="mdn", num_mixtures=4)
        states = torch.randn(4, 7, 6)
        dt = torch.ones(4, 7, 1) * 0.5
        mask = torch.ones(4, 7, 6)
        params = model(states, dt=dt, mask=mask)
        assert params["logits"].shape == (4, 4)
        assert params["loc"].shape == (4, 4, 2)
        assert model.predict_action(states, dt=dt, mask=mask).shape == (4, 2)

    def test_autoncp_policy_shape(self):
        model = LNNImitationPolicy(
            state_dim=6,
            action_dim=2,
            hidden_size=8,
            recurrent_type="autoncp",
            head_type="mdn",
            num_mixtures=3,
        )
        states = torch.randn(2, 5, 6)
        params = model(states)
        assert params["logits"].shape == (2, 3)
        assert params["loc"].shape == (2, 3, 2)


class TestGraphLNN:
    def test_graph_encoder_shape(self):
        encoder = GraphSnapshotEncoder(node_feature_size=3, hidden_size=8, output_size=6)
        node_features = torch.randn(2, 5, 4, 3)
        adjacency = torch.ones(2, 5, 4, 4)
        out = encoder(node_features, adjacency)
        assert out.shape == (2, 5, 6)

    def test_graph_lnn_predictor_shape(self):
        ds = SyntheticGraphTimeSeriesDataset(num_samples=8, seq_len=5, num_nodes=4, node_feature_size=3)
        batch, target = next(iter(create_graph_dataloaders(ds, batch_size=4)[0]))
        model = GraphLNNPredictor(node_feature_size=3, graph_feature_size=6, hidden_size=8)
        prediction = model(batch)
        assert prediction.shape == target.shape


class TestLongSequenceLiquid:
    def test_liquid_s4_block_shape(self):
        block = LiquidS4Block(input_size=4, hidden_size=8, kernel_size=3)
        x = torch.randn(2, 32, 4)
        mask = torch.ones(2, 32)
        out = block(x, mask=mask)
        assert out.shape == (2, 32, 8)

    def test_long_sequence_classifier_shape(self):
        ds = SyntheticLongSequenceDataset(num_samples=8, seq_len=48, feature_size=6, num_classes=3)
        features, target = next(iter(create_long_sequence_dataloaders(ds, batch_size=4)[0]))
        model = LongSequenceLiquidClassifier(input_size=6, num_classes=3, hidden_size=8, num_blocks=1)
        logits = model(features, mask=target["mask"])
        assert logits.shape == (4, 3)

    def test_liquid_tad_head_shape(self):
        head = LiquidTADHead(input_size=6, num_classes=4, hidden_size=8, num_blocks=1)
        x = torch.randn(2, 40, 6)
        out = head(x)
        assert out["frame_logits"].shape == (2, 40, 4)
        assert out["boundaries"].shape == (2, 40, 2)


class TestPhysicsInformedLNN:
    def test_physics_model_and_loss(self):
        ds = DampedOscillatorDataset(num_samples=8, seq_len=6, horizon=4)
        states, target = next(iter(create_physics_dataloaders(ds, batch_size=4)[0]))
        model = PhysicsInformedLNN(hidden_size=8, horizon=4, recurrent_type="cfc")
        prediction = model(states, dt=target["dt"], mask=target["mask"])
        assert prediction["params"].shape == (4, 2)
        assert prediction["rollout"].shape == (4, 4, 2)
        loss, metrics = physics_informed_loss(prediction, target)
        assert torch.isfinite(loss)
        assert metrics["rollout_loss"] >= 0.0

    def test_damped_oscillator_residual_shape(self):
        rollout = torch.randn(3, 5, 2)
        params = torch.rand(3, 2) + 0.1
        dt = torch.ones(3, 5, 1) * 0.05
        residual = damped_oscillator_residual(rollout, params, dt)
        assert residual.shape == ()
        assert torch.isfinite(residual)


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

    def test_dataset_returns_dt_mask_metadata(self):
        data = generate_sine_data(num_samples=80)
        data[10] = np.nan
        delta_t = np.linspace(0.1, 1.0, num=80, dtype=np.float32)
        mask = np.isfinite(data).astype(np.float32)
        ds = TimeSeriesDataset(data, seq_len=16, horizon=1, delta_t=delta_t, mask=mask, return_metadata=True)
        x, y, metadata = ds[0]
        assert x.shape == (16, 1)
        assert y.shape == (1,)
        assert metadata["dt"].shape == (16, 1)
        assert metadata["mask"].shape == (16, 1)
        assert torch.isfinite(x).all()
        assert metadata["mask"][10, 0] == 0.0

    def test_trainer_passes_dt_mask_metadata(self):
        data = generate_sine_data(num_samples=96)
        delta_t = np.ones_like(data, dtype=np.float32) * 0.5
        mask = np.ones_like(data, dtype=np.float32)
        mask[20:24] = 0.0
        loader = create_dataloader(
            data,
            seq_len=12,
            horizon=1,
            batch_size=8,
            delta_t=delta_t,
            mask=mask,
            shuffle=False,
        )
        model = CfCNetwork(input_size=1, hidden_size=4, output_size=1, return_sequences=False)
        trainer = Trainer(model, lr=1e-3, device="cpu", patience=1)
        history = trainer.fit(loader, num_epochs=1, verbose=False)
        preds, targets = trainer.predict(loader)
        assert history["total_epochs"] == 1
        assert preds.shape == targets.shape


class TestSyntheticMultimodalDataset:
    def test_dataset_shapes(self):
        ds = SyntheticMultimodalDataset(num_samples=10, seq_len=7, sensor_dim=3, image_size=8, text_len=5)
        sample, label = ds[0]
        assert sample["sensor"].shape == (7, 3)
        assert sample["image"].shape == (1, 8, 8)
        assert sample["tokens"].shape == (5,)
        assert label.shape == ()

    def test_dataloader_shapes(self):
        ds = SyntheticMultimodalDataset(num_samples=20, seq_len=7, sensor_dim=3, image_size=8, text_len=5)
        train_loader, _, _ = create_multimodal_dataloaders(ds, batch_size=4)
        batch, labels = next(iter(train_loader))
        assert batch["sensor"].shape == (4, 7, 3)
        assert batch["image"].shape == (4, 1, 8, 8)
        assert batch["tokens"].shape == (4, 5)
        assert labels.shape == (4,)


class TestSyntheticImitationDataset:
    def test_dataset_shapes(self):
        ds = SyntheticImitationDataset(num_samples=12, context_len=6, state_dim=6, return_metadata=True)
        states, action, metadata = ds[0]
        assert states.shape == (6, 6)
        assert action.shape == (2,)
        assert metadata["dt"].shape == (6, 1)
        assert metadata["mask"].shape == (6, 6)

    def test_dataloader_shapes(self):
        ds = SyntheticImitationDataset(num_samples=24, context_len=5, state_dim=6, return_metadata=True)
        train_loader, _, _ = create_imitation_dataloaders(ds, batch_size=4)
        states, actions, metadata = next(iter(train_loader))
        assert states.shape == (4, 5, 6)
        assert actions.shape == (4, 2)
        assert metadata["dt"].shape == (4, 5, 1)
        assert metadata["mask"].shape == (4, 5, 6)


class TestThirdStageDatasets:
    def test_graph_dataset_shapes(self):
        ds = SyntheticGraphTimeSeriesDataset(num_samples=10, seq_len=6, num_nodes=5, node_feature_size=3)
        sample, target = ds[0]
        assert sample["node_features"].shape == (6, 5, 3)
        assert sample["adjacency"].shape == (6, 5, 5)
        assert sample["dt"].shape == (6, 1)
        assert target.shape == (1,)

    def test_long_sequence_dataset_shapes(self):
        ds = SyntheticLongSequenceDataset(num_samples=10, seq_len=64, feature_size=6, num_classes=3)
        features, target = ds[0]
        assert features.shape == (64, 6)
        assert target["frame_labels"].shape == (64,)
        assert target["boundaries"].shape == (64, 2)

    def test_physics_dataset_shapes(self):
        ds = DampedOscillatorDataset(num_samples=10, seq_len=6, horizon=4)
        states, target = ds[0]
        assert states.shape == (6, 2)
        assert target["params"].shape == (2,)
        assert target["rollout"].shape == (4, 2)


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
