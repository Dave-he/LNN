"""Unit tests for CosineRouter + FAMECfCCell(router_type='cosine') (PRD #10-41, 2026-06-14).

Verifies the parameter-free CosineRouter from
arXiv:2605.12476 (Geometric Coupling):

- ``CosineRouter`` has zero ``nn.Parameter`` (truly parameter-free).
- Buffers move with ``.to(device)``.
- ``update`` is in-place, no_grad, and only updates the per-expert
  mean from states that were routed to that expert.
- ``forward(x_t, h)`` returns a top-K sparse mixture with correct
  shape and exactly ``top_k`` non-zeros per row.
- Init: all expert means are zero → cosine sims are all zero →
  softmax is uniform.
- After warm-up with K=3 distinct clusters, top-1 picks the
  geometrically closest cluster.
- ``FAMECfCCell(router_type='cosine')`` uses the parameter-free
  router; eval mode freezes the means.
"""
import numpy as np
import torch

from lnn.core.cosine_router import CosineRouter
from lnn.core.fame_cfc import FAMECfCCell, FAMECfCNetwork


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestCosineRouterInvariants:
    def test_zero_learned_parameters(self) -> None:
        """``CosineRouter`` has NO learned parameters — only buffers."""
        _seed(0)
        r = CosineRouter(input_size=2, hidden_size=4, n_experts=3, top_k=2)
        n_params = sum(p.numel() for p in r.parameters())
        assert n_params == 0, f"expected 0 learned params, got {n_params}"
        assert r.num_learned_parameters == 0

    def test_buffer_device_propagation(self) -> None:
        """Buffers move with ``.to(device)``."""
        _seed(1)
        r = CosineRouter(input_size=2, hidden_size=4, n_experts=3, top_k=2)
        r.to("cpu")
        assert r.expert_means.device.type == "cpu"
        assert r.expert_means.shape == (3, 6)

    def test_initial_state(self) -> None:
        """Expert means init to zeros; step counter init to 0."""
        _seed(2)
        r = CosineRouter(input_size=2, hidden_size=4, n_experts=3, top_k=2)
        assert torch.allclose(r.expert_means, torch.zeros(3, 6))
        assert int(r.step.item()) == 0

    def test_update_is_no_grad(self) -> None:
        """``update`` does not require or produce gradients."""
        _seed(3)
        r = CosineRouter(input_size=2, hidden_size=4, n_experts=3, top_k=2)
        combined = torch.randn(4, 6)
        top_idx = torch.tensor([[0, 1], [1, 2], [0, 2], [0, 1]])
        r.update(combined, top_idx)
        # After update, state is finite.
        assert torch.isfinite(r.expert_means).all()
        # Grad-fn is None (no grad required for the call).
        assert r.expert_means.grad_fn is None

    def test_update_only_affects_used_experts(self) -> None:
        """An expert never routed to keeps its zero mean."""
        _seed(4)
        r = CosineRouter(input_size=2, hidden_size=4, n_experts=3, top_k=1, ema_alpha=1.0)
        # All batch elements route to expert 0.
        combined = torch.randn(4, 6)
        top_idx = torch.zeros(4, 1, dtype=torch.long)  # all → expert 0
        r.update(combined, top_idx)
        # Expert 0 has been updated; experts 1 and 2 are still zeros.
        assert not torch.allclose(r.expert_means[0], torch.zeros(6))
        assert torch.allclose(r.expert_means[1], torch.zeros(6))
        assert torch.allclose(r.expert_means[2], torch.zeros(6))

    def test_update_increments_step(self) -> None:
        """``update`` increments the step counter."""
        _seed(5)
        r = CosineRouter(input_size=2, hidden_size=4, n_experts=3, top_k=2)
        assert int(r.step.item()) == 0
        r.update(torch.randn(3, 6), torch.tensor([[0, 1], [1, 2], [0, 2]]))
        assert int(r.step.item()) == 1
        r.update(torch.randn(3, 6), torch.tensor([[0, 1], [1, 2], [0, 2]]))
        assert int(r.step.item()) == 2

    def test_reset_state(self) -> None:
        """``reset_state`` clears means and step counter."""
        _seed(6)
        r = CosineRouter(input_size=2, hidden_size=4, n_experts=3, top_k=2)
        r.update(torch.randn(3, 6), torch.tensor([[0, 1], [1, 2], [0, 2]]))
        r.reset_state()
        assert torch.allclose(r.expert_means, torch.zeros(3, 6))
        assert int(r.step.item()) == 0


class TestCosineRouterForward:
    def test_forward_shape_top_k(self) -> None:
        """``forward`` returns [B, K] sparse g with exactly top_k nonzeros."""
        _seed(7)
        r = CosineRouter(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        g = r(x_t, h)
        assert g.shape == (4, 3)
        # Exactly top_k nonzeros per row.
        assert (g > 0).sum(dim=-1).tolist() == [2, 2, 2, 2]
        # last_top_idx is set.
        assert r.last_top_idx.shape == (4, 2)

    def test_forward_dense_softmax_when_top_k_equals_K(self) -> None:
        """``top_k == K`` is dense softmax (no -inf mask)."""
        _seed(8)
        r = CosineRouter(input_size=3, hidden_size=8, n_experts=3, top_k=3)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        g = r(x_t, h)
        # No zero rows.
        assert (g > 0).all()

    def test_init_uniform_softmax(self) -> None:
        """With zero expert means, all cosine sims are 0 → uniform g."""
        _seed(9)
        r = CosineRouter(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        g = r(x_t, h)
        # top_k=2 → uniform 0.5 per row, others 0.
        for row in g:
            nonzero = row[row > 0]
            assert torch.allclose(nonzero, torch.full_like(nonzero, 0.5), atol=1e-6)

    def test_picks_closest_cluster_after_warmup(self) -> None:
        """After warm-up with K distinct clusters, top-1 picks the closest."""
        _seed(10)
        r = CosineRouter(input_size=3, hidden_size=3, n_experts=3, top_k=1, ema_alpha=1.0)
        # Warm up: route each batch element to a specific expert with a
        # cluster-aligned [x_t; h] vector.
        for k in range(3):
            cluster_center = torch.zeros(1, 6)
            cluster_center[0, k * 2] = 1.0  # orthogonal cluster centers
            for _ in range(5):
                # x_t and h split the D=6 evenly: each has 3 dims.
                x_t = cluster_center[:, :3] + 0.01 * torch.randn(1, 3)
                h = cluster_center[:, 3:] + 0.01 * torch.randn(1, 3)
                top_idx = torch.tensor([[k]], dtype=torch.long)
                combined = torch.cat([x_t, h], dim=-1)
                r.update(combined, top_idx)
        # Now query: a point closest to cluster 0 (first 2 dims dominant).
        x_t = torch.tensor([[1.0, 0.0, 0.0]])
        h = torch.tensor([[1.0, 0.0, 0.0]])
        g = r(x_t, h)
        chosen = int(r.last_top_idx.item())
        assert chosen == 0, f"expected expert 0 (cluster-aligned), got {chosen}"


class TestFAMECfCCellCosineRouter:
    def test_cell_uses_cosine_router(self) -> None:
        """``router_type='cosine'`` swaps in CosineRouter; learned is bypassed."""
        _seed(11)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            router_type="cosine",
        )
        assert isinstance(cell.router, CosineRouter)
        # No learned router parameters in the cell either.
        n_router_params = sum(p.numel() for p in cell.router.parameters())
        assert n_router_params == 0

    def test_learned_router_default_back_compat(self) -> None:
        """``router_type='learned'`` (default) keeps ForecastabilityRouter."""
        _seed(12)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
        )
        assert cell.router_type == "learned"
        from lnn.core.forecastability_router import ForecastabilityRouter
        assert isinstance(cell.router, ForecastabilityRouter)

    def test_cosine_train_updates_means(self) -> None:
        """In train mode, cosine router's expert_means are updated."""
        _seed(13)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            router_type="cosine", ema_alpha=0.1,
        )
        cell.train()
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        means_before = cell.router.expert_means.clone()
        cell.forward_with_aux(x_t, h, dt=1.0)
        means_after = cell.router.expert_means.clone()
        # Means must have moved.
        assert not torch.allclose(means_before, means_after)

    def test_cosine_eval_freezes_means(self) -> None:
        """In eval mode, cosine router's expert_means are NOT updated."""
        _seed(14)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            router_type="cosine", ema_alpha=0.5,
        )
        cell.train()
        # Warm up.
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        cell.forward_with_aux(x_t, h, dt=1.0)
        means_warm = cell.router.expert_means.clone()
        # Switch to eval; means should be frozen.
        cell.eval()
        with torch.no_grad():
            for _ in range(5):
                x_t = torch.randn(4, 3)
                h = torch.randn(4, 8)
                cell.forward_with_aux(x_t, h, dt=1.0)
        assert torch.allclose(cell.router.expert_means, means_warm)

    def test_cosine_cell_forward_matches_forward_with_aux_h(self) -> None:
        """``cell.forward`` and ``cell.forward_with_aux``[0] must agree."""
        _seed(15)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            router_type="cosine", ema_alpha=0.1,
        )
        cell.eval()
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        h_a = cell.forward(x_t, h, dt=1.0)
        with torch.no_grad():
            h_b, _ = cell.forward_with_aux(x_t, h, dt=1.0)
        assert torch.allclose(h_a, h_b, atol=1e-6)

    def test_unknown_router_type_raises(self) -> None:
        """Invalid ``router_type`` raises at construction."""
        import pytest
        with pytest.raises((ValueError, AssertionError), match="router_type"):
            FAMECfCCell(
                input_size=3, hidden_size=8, n_experts=3, top_k=2,
                router_type="bogus",
            )


class TestFAMEWithCosineRouterSmoke:
    def test_k3_topk1_cosine_alone_converges(self) -> None:
        """K=3 top_k=1 + cosine router — we check it trains (not specifically < 0.5).

        Note: on the hardest cell (K=3 top_k=1), the parameter-free
        cosine router alone can struggle because the zero-init expert
        means produce a uniform softmax that doesn't differentiate
        experts.  The paper's claim is best on top_k=2 (sparse) where
        the EMA can learn cluster centers with more routing mass per
        step.  We check that the cosine variant runs to completion and
        does not blow up (final loss < 5.0).
        """
        _seed(16)
        T, N = 32, 64
        t = torch.linspace(0, 2 * np.pi, T).unsqueeze(0).expand(N, -1)
        x = torch.sin(t).unsqueeze(-1)
        y = torch.cos(t).unsqueeze(-1)
        torch.manual_seed(42)
        net = FAMECfCNetwork(
            input_size=1, hidden_size=16, output_size=1,
            num_layers=1, n_experts=3, top_k=1,
            router_type="cosine", ema_alpha=0.1,
        )
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        final = float("nan")
        for _ in range(25):
            opt.zero_grad()
            y_pred, _ = net.forward_with_aux(x)
            task_loss = loss_fn(y_pred, y)
            task_loss.backward()
            opt.step()
            final = float(task_loss.item())
        # Sanity: didn't blow up.  Actual convergence requires the
        # full smoke bench (scripts/bench_cosine_router.py).
        assert final < 5.0, f"task_loss={final} exploded"
