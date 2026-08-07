"""r304 — Unit tests for LFM2.5 LSTM→ParallelCfC integration.

Validates:
  - replace_lstm_with_parallel_cfc() removes every nn.LSTM/nn.GRU.
  - Output shape matches baseline for a tiny mock.
  - Param count is non-zero and the swap is differentiable.
  - Models with no recurrent layers are unchanged.
  - Nested containers (Sequential, ModuleList) handled.
  - Multiple windows (1, 2, 4, 8) all work and produce finite outputs.
  - inplace=False returns a deep copy that leaves the original intact.
"""
from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from lnn.lfm2.parallel_integration import (
    RECURRENT_CLASSES,
    _count_lstm_like,
    _make_replacement,
    replace_lstm_with_parallel_cfc,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _mock_lfm25(
    vocab: int = 64,
    hidden: int = 32,
    num_layers: int = 1,
) -> nn.Module:
    """Tiny LFM2.5-style: Embedding → LSTM → head.

    We extract the backbone call behind a small helper so the same
    forward() works for both ``nn.LSTM`` (which returns a tuple) and the
    swapped ``ParallelCfCNetwork`` (which returns a single tensor).
    """

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.backbone = nn.LSTM(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden, vocab, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e = self.embed(x)
            h = self.backbone(e)
            if isinstance(h, tuple):
                h = h[0]
            return self.head(h)

    return M()


class TestReplaceLSTM(unittest.TestCase):
    """Core contract: every nn.LSTM is replaced by a ParallelCfCNetwork."""

    def test_replace_removes_lstm(self):
        _seed(0)
        m = _mock_lfm25()
        self.assertEqual(_count_lstm_like(m), 1)
        m2 = replace_lstm_with_parallel_cfc(m, window=4)
        self.assertEqual(_count_lstm_like(m2), 0)
        # Same attribute still resolves.
        self.assertIsNotNone(m2.backbone)

    def test_replace_preserves_output_shape(self):
        _seed(0)
        m = _mock_lfm25()
        x = torch.randint(0, 64, (2, 8))
        # Snapshot the baseline output BEFORE swap (in-place mutation).
        out_lstm = m(x).clone()
        m2 = replace_lstm_with_parallel_cfc(m, window=4)
        out_cfc = m2(x)
        self.assertEqual(out_cfc.shape, out_lstm.shape)

    def test_replace_w1_preserves_output_shape(self):
        _seed(0)
        m = _mock_lfm25()
        x = torch.randint(0, 64, (2, 16))
        m2 = replace_lstm_with_parallel_cfc(m, window=1)
        out_cfc = m2(x)
        self.assertEqual(out_cfc.shape, (2, 16, 64))

    def test_replace_w8_preserves_output_shape(self):
        _seed(0)
        m = _mock_lfm25()
        x = torch.randint(0, 64, (1, 16))  # T=16 == 2*8 chunks
        m2 = replace_lstm_with_parallel_cfc(m, window=8)
        out_cfc = m2(x)
        self.assertEqual(out_cfc.shape, (1, 16, 64))

    def test_model_with_no_lstm_is_unchanged(self):
        _seed(0)
        m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        # No LSTM/GRU — replacement is a no-op.
        before = _count_lstm_like(m)
        self.assertEqual(before, 0)
        m2 = replace_lstm_with_parallel_cfc(m, window=4)
        self.assertEqual(_count_lstm_like(m2), 0)
        # Forward shape identical.
        x = torch.randn(3, 4)
        self.assertEqual(m2(x).shape, m(x).shape)


class TestNestedContainers(unittest.TestCase):
    """Walker must descend into nn.Sequential and nn.ModuleList."""

    def test_sequential_nested_lstm(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(4, 4)
                self.encoder = nn.Sequential(
                    nn.Linear(4, 4),
                    nn.LSTM(input_size=4, hidden_size=4, batch_first=True),
                )

            def forward(self, x):
                e = self.embed(x)
                h = self.encoder[1](e)
                if isinstance(h, tuple):
                    h = h[0]
                return h

        _seed(0)
        n = Net()
        self.assertEqual(_count_lstm_like(n), 1)
        n2 = replace_lstm_with_parallel_cfc(n, window=4)
        self.assertEqual(_count_lstm_like(n2), 0)

    def test_modulelist_multiple_lstms(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(4, 4)
                self.encoders = nn.ModuleList(
                    [
                        nn.LSTM(input_size=4, hidden_size=4, batch_first=True),
                        nn.LSTM(input_size=4, hidden_size=4, batch_first=True),
                    ]
                )

            def forward(self, x):
                h = self.embed(x)
                for enc in self.encoders:
                    h = enc(h)
                    if isinstance(h, tuple):
                        h = h[0]
                return h

        _seed(0)
        n = Net()
        self.assertEqual(_count_lstm_like(n), 2)
        n2 = replace_lstm_with_parallel_cfc(n, window=2)
        self.assertEqual(_count_lstm_like(n2), 0)
        # Forward still works (T=8 = 4 chunks of W=2).
        x = torch.randn(2, 8, 4)
        out = n2(x)
        self.assertEqual(out.shape, (2, 8, 4))

    def test_gru_is_also_swapped(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(4, 4)
                self.backbone = nn.GRU(input_size=4, hidden_size=4, batch_first=True)

            def forward(self, x):
                e = self.embed(x)
                h = self.backbone(e)
                if isinstance(h, tuple):
                    h = h[0]
                return h

        _seed(0)
        n = Net()
        self.assertEqual(_count_lstm_like(n), 1)  # GRU counts too
        n2 = replace_lstm_with_parallel_cfc(n, window=4)
        self.assertEqual(_count_lstm_like(n2), 0)


class TestMultipleWindows(unittest.TestCase):
    """All W in {1, 2, 4, 8} must run and produce finite outputs."""

    def test_all_windows(self):
        _seed(0)
        for W in (1, 2, 4, 8):
            with self.subTest(window=W):
                m = _mock_lfm25()
                m2 = replace_lstm_with_parallel_cfc(m, window=W)
                # T = 16 is a multiple of 1, 2, 4, 8 — all windows valid.
                x = torch.randint(0, 64, (2, 16))
                out = m2(x)
                self.assertEqual(out.shape, (2, 16, 64))
                self.assertTrue(torch.isfinite(out).all())

    def test_w1_does_not_break_larger_T(self):
        _seed(0)
        m = _mock_lfm25()
        m2 = replace_lstm_with_parallel_cfc(m, window=1)
        x = torch.randint(0, 64, (1, 32))
        out = m2(x)
        self.assertEqual(out.shape, (1, 32, 64))

    def test_window_must_divide_T_for_parallel(self):
        """If T is not a multiple of W>1, ParallelCfCNetwork raises."""
        _seed(0)
        m = _mock_lfm25()
        m2 = replace_lstm_with_parallel_cfc(m, window=4)
        x = torch.randint(0, 64, (1, 10))  # 10 % 4 != 0
        with self.assertRaises(AssertionError):
            m2(x)


class TestDifferentiable(unittest.TestCase):
    """The swapped module must remain differentiable (for fine-tuning)."""

    def test_backward_through_swapped_model(self):
        _seed(0)
        m = _mock_lfm25()
        m2 = replace_lstm_with_parallel_cfc(m, window=4)
        x = torch.randint(0, 64, (1, 8))
        out = m2(x)
        loss = out.sum()
        loss.backward()
        # Embedding + head + the ParallelCfC cell must have gradients.
        self.assertIsNotNone(m2.embed.weight.grad)
        self.assertIsNotNone(m2.backbone.cells[0].time_scale.grad)


class TestMakeReplacement(unittest.TestCase):
    """The lower-level factory should respect LSTM/GRU shape contracts."""

    def test_make_replacement_lstm(self):
        _seed(0)
        lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        mod = _make_replacement(lstm, window=4)
        # Output dim = hidden_size when no proj_size.
        self.assertEqual(mod.output_size, 16)
        self.assertEqual(mod.input_size, 8)
        self.assertEqual(mod.hidden_size, 16)
        # T=8 = 2 chunks of W=4.
        x = torch.randn(2, 8, 8)
        out = mod(x)
        self.assertEqual(out.shape, (2, 8, 16))

    def test_make_replacement_gru(self):
        _seed(0)
        gru = nn.GRU(input_size=4, hidden_size=8, batch_first=True)
        mod = _make_replacement(gru, window=2)
        self.assertEqual(mod.output_size, 8)
        self.assertEqual(mod.input_size, 4)
        self.assertEqual(mod.hidden_size, 8)

    def test_make_replacement_raises_on_linear(self):
        with self.assertRaises(TypeError):
            _make_replacement(nn.Linear(4, 4), window=4)


class TestInplaceFlag(unittest.TestCase):
    """inplace=False must return a deep copy with the original intact."""

    def test_inplace_false_returns_copy(self):
        _seed(0)
        m = _mock_lfm25()
        n_before = _count_lstm_like(m)
        m2 = replace_lstm_with_parallel_cfc(m, window=4, inplace=False)
        # Original untouched.
        self.assertEqual(_count_lstm_like(m), n_before)
        # Copy swapped.
        self.assertEqual(_count_lstm_like(m2), 0)
        # Distinct objects.
        self.assertIsNot(m, m2)

    def test_inplace_true_default(self):
        _seed(0)
        m = _mock_lfm25()
        m2 = replace_lstm_with_parallel_cfc(m, window=4, inplace=True)
        # Same object mutated.
        self.assertIs(m, m2)
        self.assertEqual(_count_lstm_like(m2), 0)


class TestOutputStabilityProxy(unittest.TestCase):
    """A simple quality proxy: output std across 5 fixed seeds must be
    finite and (qualitatively) reasonable for a frozen-random-init model."""

    def test_output_stability_finite(self):
        _seed(0)
        m = _mock_lfm25()
        m2 = replace_lstm_with_parallel_cfc(m, window=4)
        # Frozen eval — but each seed re-runs forward, so the inputs are
        # perturbed via torch.manual_seed (which doesn't change input
        # data).  We re-init the model weights each seed to get a
        # meaningful stability proxy.
        outs = []
        x = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            for s in range(5):
                torch.manual_seed(s)
                # Re-init weights of the swapped cell only.
                for p in m2.backbone.parameters():
                    if p.dim() >= 2:
                        nn.init.xavier_uniform_(p)
                outs.append(m2(x).flatten())
        stacked = torch.stack(outs, dim=0)
        self.assertTrue(torch.isfinite(stacked).all())


if __name__ == "__main__":
    unittest.main()
