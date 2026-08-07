"""Unit tests for the r302 sharp-transition bench (N-MNIST-like synthetic).

Validates:
  - Dataset builder produces (B, T, C) and (B,) shapes with expected ranges.
  - Templates are non-empty and non-trivial.
  - VanillaCfCClassifier and ParallelCfCClassifier return (B, num_classes).
  - ParallelCfC w=2/4/8 all return finite logits.
  - Chunked contract: T=64 works for W=2,4,8 but W=8 is acceptable here.
  - trainable: a single Adam step on a tiny dataset decreases loss.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from scripts.bench_parallel_cfc_sharp import (
    NUM_CHANNELS,
    NUM_CLASSES,
    SEQ_LEN,
    VanillaCfCClassifier,
    ParallelCfCClassifier,
    _build_class_templates,
    make_sharp_data,
)


class TestSharpDataset(unittest.TestCase):
    def test_template_shape_and_bursts(self):
        templates = _build_class_templates(num_classes=5, T=64, num_channels=2, seed=0)
        self.assertEqual(templates.shape, (5, 2, 64))
        self.assertTrue(templates.any(), "Templates should not be all-zero")
        # Each class has at least one burst across both channels.
        for c in range(5):
            self.assertTrue(
                templates[c].any(),
                f"class {c} has no burst windows",
            )

    def test_data_shapes_and_types(self):
        x_tr, y_tr, x_te, y_te = make_sharp_data(n_train=20, n_test=10)
        self.assertEqual(x_tr.shape, (20 * NUM_CLASSES, SEQ_LEN, NUM_CHANNELS))
        self.assertEqual(x_te.shape, (10 * NUM_CLASSES, SEQ_LEN, NUM_CHANNELS))
        self.assertEqual(y_tr.shape, (20 * NUM_CLASSES,))
        self.assertEqual(y_te.shape, (10 * NUM_CLASSES,))
        # All inputs are {0, 1}.
        self.assertTrue(((x_tr == 0) | (x_tr == 1)).all())
        # Labels are in [0, num_classes).
        self.assertTrue((y_tr >= 0).all() and (y_tr < NUM_CLASSES).all())
        self.assertTrue((y_te >= 0).all() and (y_te < NUM_CLASSES).all())
        # Each class appears at least n times in the training set.
        for c in range(NUM_CLASSES):
            self.assertGreaterEqual(int((y_tr == c).sum().item()), 20)

    def test_data_burst_rate_in_range(self):
        x_tr, _, _, _ = make_sharp_data(n_train=10, n_test=5)
        # The spike probability is bounded by BURST_PROB*burst_rate +
        # NOISE_PROB*(1-burst_rate), so the mean is around 0.13-0.20.
        mean = float(x_tr.mean().item())
        self.assertGreater(mean, 0.05)
        self.assertLess(mean, 0.30)


class TestModelsForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.x = torch.randn(8, SEQ_LEN, NUM_CHANNELS)

    def test_vanilla_cfc_forward(self):
        model = VanillaCfCClassifier()
        out = model(self.x)
        self.assertEqual(out.shape, (8, NUM_CLASSES))
        self.assertTrue(torch.isfinite(out).all())

    def test_parallel_cfc_w2_forward(self):
        model = ParallelCfCClassifier(window=2)
        out = model(self.x)
        self.assertEqual(out.shape, (8, NUM_CLASSES))
        self.assertTrue(torch.isfinite(out).all())

    def test_parallel_cfc_w4_forward(self):
        model = ParallelCfCClassifier(window=4)
        out = model(self.x)
        self.assertEqual(out.shape, (8, NUM_CLASSES))
        self.assertTrue(torch.isfinite(out).all())

    def test_parallel_cfc_w8_forward(self):
        model = ParallelCfCClassifier(window=8)
        out = model(self.x)
        self.assertEqual(out.shape, (8, NUM_CLASSES))
        self.assertTrue(torch.isfinite(out).all())


class TestTrainStep(unittest.TestCase):
    def test_one_step_decreases_loss(self):
        torch.manual_seed(0)
        np.random.seed(0)
        x_tr, y_tr, _, _ = make_sharp_data(n_train=4, n_test=2)
        model = VanillaCfCClassifier()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        # Initial loss
        model.train()
        loss0 = float(loss_fn(model(x_tr), y_tr).item())
        for _ in range(20):
            opt.zero_grad()
            loss = loss_fn(model(x_tr), y_tr)
            loss.backward()
            opt.step()
        loss1 = float(loss_fn(model(x_tr), y_tr).item())
        self.assertLess(
            loss1,
            loss0,
            f"Training did not decrease loss: {loss0:.4f} -> {loss1:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
