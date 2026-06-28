"""Unit tests for SoftNeuronAttentionCfCCell (round 264)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.soft_neuron_attention_cfc import SoftNeuronAttentionCfCCell


class TestSoftNeuronAttentionCfCCellBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        self.assertEqual(cell.input_size, 4)
        self.assertEqual(cell.hidden_size, 8)
        self.assertEqual(cell.l1_lambda, 0.01)

    def test_init_l1_lambda_validation(self):
        with self.assertRaises(ValueError):
            SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8, l1_lambda=-0.1)

    def test_init_hidden_size_validation(self):
        with self.assertRaises(ValueError):
            SoftNeuronAttentionCfCCell(input_size=4, hidden_size=1)

    def test_has_required_parameters(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        for name in [
            "W_rec", "neighbor_logits", "log_tau_attn",
            "tau_per_neuron", "alpha_per_neuron",
            "bias_per_neuron", "input_strength_per_neuron", "W_in",
        ]:
            self.assertTrue(hasattr(cell, name), f"missing param: {name}")


class TestSoftNeuronAttentionCfCCellForward(unittest.TestCase):

    def test_forward_shape(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        x = torch.randn(2, 16, 4)
        out, h = cell(x)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertEqual(h.shape, (2, 8))
        del out, h

    def test_forward_with_h0(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        x = torch.randn(2, 16, 4)
        h0 = torch.randn(2, 8)
        out, h = cell(x, h0=h0)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertFalse(torch.allclose(h, h0))
        del out

    def test_forward_aux_dict_keys(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        x = torch.randn(2, 16, 4)
        _, _, aux = cell(x, return_aux=True)
        for key in [
            "attention", "tau_attn", "attention_entropy_mean",
            "attention_entropy_std", "attention_max_weight",
            "attention_sparsity", "sparsity_loss_value",
            "tau_summary", "alpha_mean", "alpha_std",
        ]:
            self.assertIn(key, aux, f"missing aux key: {key}")

    def test_output_finite(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        x = torch.randn(2, 16, 4)
        out, h = cell(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(h).all())
        del out, h

    def test_gradients_flow_including_neighbor_logits(self):
        """KEY DIFFERENCE from r263: neighbor_logits IS grad-trained here."""
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        x = torch.randn(2, 16, 4)
        target = torch.randn(2, 16, 8)
        out, _ = cell(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        # neighbor_logits must receive a finite grad.
        self.assertIsNotNone(cell.neighbor_logits.grad)
        self.assertTrue(torch.isfinite(cell.neighbor_logits.grad).all())
        # log_tau_attn must also receive a finite grad.
        self.assertIsNotNone(cell.log_tau_attn.grad)
        self.assertTrue(torch.isfinite(cell.log_tau_attn.grad).all())
        # Other params also get grad.
        for name in ["W_rec", "tau_per_neuron", "alpha_per_neuron",
                     "bias_per_neuron", "input_strength_per_neuron"]:
            p = getattr(cell, name)
            self.assertIsNotNone(p.grad, f"no grad for {name}")


class TestSoftNeuronAttentionCfCCellAttention(unittest.TestCase):

    def test_attention_row_stochastic(self):
        """Each row of attention must sum to 1."""
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        alpha = cell.get_attention()
        row_sums = alpha.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(8), atol=1e-5))

    def test_attention_all_positive(self):
        """Softmax outputs must be positive."""
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        alpha = cell.get_attention()
        self.assertTrue((alpha > 0).all())

    def test_tau_attn_positive(self):
        """τ_attn must always be positive (softplus + clamp)."""
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        with torch.no_grad():
            cell.log_tau_attn.fill_(-100.0)  # very negative
        tau = cell.get_tau_attn()
        self.assertGreater(float(tau.item()), 0.0)

    def test_attention_entropy_in_range(self):
        """Per-row entropy must be in (0, log(d_h))."""
        d_h = 8
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=d_h)
        entropy = cell.get_attention_entropy()
        self.assertTrue((entropy > 0).all())
        self.assertTrue((entropy <= math.log(d_h) + 1e-5).all())

    def test_low_tau_attn_produces_peaked_attention(self):
        """Lower τ_attn → sharper (lower entropy) attention."""
        cell_sharp = SoftNeuronAttentionCfCCell(
            input_size=4, hidden_size=8, init_tau_attn=0.1, seed=42
        )
        cell_soft = SoftNeuronAttentionCfCCell(
            input_size=4, hidden_size=8, init_tau_attn=10.0, seed=42
        )
        # Same neighbor_logits (same seed, same init).
        cell_soft.neighbor_logits.data = cell_sharp.neighbor_logits.data.clone()
        h_sharp = cell_sharp.get_attention_entropy().mean().item()
        h_soft = cell_soft.get_attention_entropy().mean().item()
        self.assertLess(h_sharp, h_soft)


class TestSoftNeuronAttentionCfCCellSparsity(unittest.TestCase):

    def test_sparsity_loss_zero_when_lambda_zero(self):
        cell = SoftNeuronAttentionCfCCell(
            input_size=4, hidden_size=8, l1_lambda=0.0
        )
        loss = cell.sparsity_loss()
        self.assertEqual(float(loss.item()), 0.0)

    def test_sparsity_loss_positive_when_lambda_positive(self):
        cell = SoftNeuronAttentionCfCCell(
            input_size=4, hidden_size=8, l1_lambda=0.1
        )
        loss = cell.sparsity_loss()
        self.assertGreater(float(loss.item()), 0.0)

    def test_sparsity_loss_grad_to_neighbor_logits(self):
        """Sparsity loss must flow grad to neighbor_logits."""
        cell = SoftNeuronAttentionCfCCell(
            input_size=4, hidden_size=8, l1_lambda=0.1
        )
        cell.sparsity_loss().backward()
        self.assertIsNotNone(cell.neighbor_logits.grad)
        self.assertTrue(torch.isfinite(cell.neighbor_logits.grad).all())


class TestSoftNeuronAttentionCfCCellDiagnostics(unittest.TestCase):

    def test_get_tau_bounded(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        tau = cell.get_tau()
        self.assertTrue((tau >= cell.tau_min).all())
        self.assertTrue((tau <= cell.tau_max).all())

    def test_get_alpha_bounded(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        with torch.no_grad():
            cell.alpha_per_neuron.fill_(100.0)
        alpha = cell.get_alpha()
        self.assertTrue((alpha <= cell.alpha_max + 1e-6).all())
        self.assertEqual(float(alpha.max().item()), cell.alpha_max)

    def test_attention_max_weight_in_range(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        m = cell.attention_max_weight()
        self.assertGreaterEqual(m, 1.0 / 8.0)  # at least uniform
        self.assertLessEqual(m, 1.0)


class TestSoftNeuronAttentionCfCCellBatchIndependence(unittest.TestCase):

    def test_batch_elems_independent(self):
        cell = SoftNeuronAttentionCfCCell(input_size=4, hidden_size=8)
        x1 = torch.zeros(1, 16, 4)
        x2 = torch.ones(1, 16, 4)
        x = torch.cat([x1, x2], dim=0)
        out, _ = cell(x)
        self.assertFalse(torch.allclose(out[0], out[1]))


if __name__ == "__main__":
    unittest.main()
