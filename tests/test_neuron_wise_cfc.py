"""Unit tests for NeuronWiseCfCCell (round 263)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.neuron_wise_cfc import (
    NeuronWiseCfCCell,
    sparse_topk_mask,
)


class TestSparseTopkMask(unittest.TestCase):

    def test_shape_preserved(self):
        logits = torch.randn(8, 8)
        m = sparse_topk_mask(logits, density=0.5)
        self.assertEqual(m.shape, (8, 8))

    def test_topk_count_per_row(self):
        n = 10
        k = 3
        density = k / n
        logits = torch.randn(n, n)
        m = sparse_topk_mask(logits, density=density)
        # Each row should have exactly k ones (k = round(density*n))
        row_sums = m.sum(dim=-1)
        self.assertTrue(torch.all(row_sums == k))

    def test_self_edge_always_kept(self):
        logits = torch.full((5, 5), -1e9)  # everything very negative
        m = sparse_topk_mask(logits, density=0.4)
        diag = torch.diagonal(m)
        self.assertTrue(torch.all(diag == 1.0))

    def test_density_extremes(self):
        logits = torch.randn(6, 6)
        m_full = sparse_topk_mask(logits, density=1.0)
        self.assertTrue(torch.all(m_full == 1.0))
        m_min = sparse_topk_mask(logits, density=0.01)
        # Density < 1/n may still keep 1 (top-1) per row.
        row_sums_min = m_min.sum(dim=-1)
        self.assertTrue(torch.all(row_sums_min == 1))


class TestNeuronWiseCfCCellBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8)
        self.assertEqual(cell.input_size, 4)
        self.assertEqual(cell.hidden_size, 8)
        self.assertEqual(cell.density, 0.3)

    def test_init_density_validation(self):
        with self.assertRaises(ValueError):
            NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.0)
        with self.assertRaises(ValueError):
            NeuronWiseCfCCell(input_size=4, hidden_size=8, density=1.5)

    def test_init_hidden_size_validation(self):
        with self.assertRaises(ValueError):
            NeuronWiseCfCCell(input_size=4, hidden_size=1)

    def test_has_required_parameters(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8)
        for name in [
            "W_rec", "neighbor_logits", "tau_per_neuron",
            "alpha_per_neuron", "bias_per_neuron",
            "input_strength_per_neuron", "W_in",
        ]:
            self.assertTrue(hasattr(cell, name), f"missing param: {name}")


class TestNeuronWiseCfCCellForward(unittest.TestCase):

    def test_forward_shape(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        out, h = cell(x)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertEqual(h.shape, (2, 8))
        del out, h  # silence unused

    def test_forward_with_h0(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        h0 = torch.randn(2, 8)
        out, h = cell(x, h0=h0)
        self.assertEqual(out.shape, (2, 16, 8))
        # Final state should differ from h0 because the input drives update.
        self.assertFalse(torch.allclose(h, h0))
        del out

    def test_forward_aux_dict_keys(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        out, h, aux = cell(x, return_aux=True)
        for key in [
            "mask", "neighborhood_density", "neighborhood_asymmetry",
            "tau_summary", "alpha_mean", "alpha_std",
        ]:
            self.assertIn(key, aux, f"missing aux key: {key}")
        for k in ["mean", "std", "min", "max", "cv"]:
            self.assertIn(k, aux["tau_summary"])

    def test_output_finite(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        out, h = cell(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(h).all())
        del out, h

    def test_gradients_flow(self):
        """All *gradient-trained* parameters must receive a finite grad.

        Note: ``neighbor_logits`` is the *top-k* operator's input and
        is intentionally NOT trained via gradient (topk is not
        differentiable). It can be learned via evolutionary search,
        REINFORCE, or replaced with a soft-mask approximation if
        structure learning is desired.
        """
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4, requires_grad=False)
        target = torch.randn(2, 16, 8)
        out, _ = cell(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        grad_trained = {
            "W_rec", "tau_per_neuron", "alpha_per_neuron",
            "bias_per_neuron", "input_strength_per_neuron", "W_in.weight",
        }
        for name, p in cell.named_parameters():
            if name not in grad_trained:
                # neighbor_logits and similar non-differentiable params.
                continue
            self.assertIsNotNone(p.grad, f"no grad for {name}")
            self.assertTrue(torch.isfinite(p.grad).all(),
                            f"non-finite grad for {name}")

    def test_neighbor_logits_not_grad_trained(self):
        """Design choice: neighbor_logits is structural, not learned via grad."""
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        target = torch.randn(2, 16, 8)
        out, _ = cell(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        # neighbor_logits.grad should be None (topk is not differentiable).
        self.assertIsNone(cell.neighbor_logits.grad)


class TestNeuronWiseCfCCellDiagnostics(unittest.TestCase):

    def test_get_tau_bounded(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        tau = cell.get_tau()
        self.assertTrue((tau >= cell.tau_min).all())
        self.assertTrue((tau <= cell.tau_max).all())

    def test_get_alpha_bounded(self):
        # Push alpha logits very high; expect clamp.
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        with torch.no_grad():
            cell.alpha_per_neuron.fill_(100.0)
        alpha = cell.get_alpha()
        self.assertTrue((alpha <= cell.alpha_max + 1e-6).all())
        self.assertEqual(float(alpha.max().item()), cell.alpha_max)

    def test_neighborhood_density_off_diagonal(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        d = cell.neighborhood_density()
        # Should be at most density (off-diagonal contribution).
        self.assertLessEqual(d, 0.5 + 1e-6)

    def test_neighborhood_asymmetry_in_range(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        a = cell.neighborhood_asymmetry()
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(a, 1.0)

    def test_density_1_is_fully_connected(self):
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=1.0)
        mask = cell.get_neighborhood_mask()
        self.assertTrue(torch.all(mask == 1.0))


class TestNeuronWiseCfCCellBatchIndependence(unittest.TestCase):

    def test_batch_elems_independent(self):
        """Two batch elements with different inputs should produce different outputs."""
        cell = NeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x1 = torch.zeros(1, 16, 4)
        x2 = torch.ones(1, 16, 4)
        x = torch.cat([x1, x2], dim=0)
        out, _ = cell(x)
        # First and second batch elements should differ.
        self.assertFalse(torch.allclose(out[0], out[1]))


if __name__ == "__main__":
    unittest.main()
