"""Unit tests for round 92 temporal dropout helper (PRD #10-54, response
to arXiv:2605.27467, Thu/Oo/Supnithi May 2026).

Verifies:
- p=0 returns unchanged
- p=1 returns all zeros
- p=0.5 returns ~half zeros
- mask is reproducible with seed
- dropout_mask helper works
- ValueError on p out of range
- input_dropout (round 93) and apply_input_dropout_to_input work
"""
import torch

from lnn.core.temporal_dropout import (
    apply_input_dropout_to_input,
    dropout_mask,
    input_dropout,
    temporal_dropout,
)


class TestTemporalDropout:
    def test_p0_returns_unchanged(self) -> None:
        t = torch.linspace(0, 1, 10)
        y = torch.sin(2 * 3.14159 * t)
        t_out, y_out = temporal_dropout(t, y, p=0.0)
        assert torch.equal(t_out, t)
        assert torch.equal(y_out, y)

    def test_p1_returns_all_zeros(self) -> None:
        t = torch.linspace(0, 1, 10)
        y = torch.ones(10) * 5.0
        _, y_out = temporal_dropout(t, y, p=1.0)
        assert torch.all(y_out == 0.0)

    def test_p05_approximate_half_zeros(self) -> None:
        """At p=0.5, expect ~50% zeros (within 30-70% range)."""
        torch.manual_seed(42)
        t = torch.linspace(0, 1, 1000)
        y = torch.ones(1000) * 7.0
        _, y_out = temporal_dropout(t, y, p=0.5)
        n_zero = (y_out == 0.0).sum().item()
        # Expect 500 ± some noise; 30-70% range is safe.
        assert 300 < n_zero < 700, f"expected ~500 zeros, got {n_zero}"

    def test_reproducible_with_seed(self) -> None:
        """Same seed → same mask."""
        t = torch.linspace(0, 1, 50)
        y = torch.ones(50)
        _, y_a = temporal_dropout(t, y, p=0.3, seed=0)
        _, y_b = temporal_dropout(t, y, p=0.3, seed=0)
        assert torch.equal(y_a, y_b)

    def test_different_seed_different_mask(self) -> None:
        """Different seed → different mask (with high probability)."""
        t = torch.linspace(0, 1, 50)
        y = torch.ones(50)
        _, y_a = temporal_dropout(t, y, p=0.3, seed=0)
        _, y_b = temporal_dropout(t, y, p=0.3, seed=1)
        assert not torch.equal(y_a, y_b)

    def test_invalid_p_raises(self) -> None:
        t = torch.linspace(0, 1, 10)
        y = torch.ones(10)
        for bad_p in (-0.1, 1.1, 2.0):
            try:
                temporal_dropout(t, y, p=bad_p)
                assert False, f"should have raised for p={bad_p}"
            except ValueError:
                pass

    def test_preserves_t(self) -> None:
        """t is always returned unchanged, only y is masked."""
        t = torch.linspace(0, 1, 10)
        y = torch.ones(10)
        t_out, _ = temporal_dropout(t, y, p=0.5, seed=0)
        assert torch.equal(t_out, t)


class TestDropoutMask:
    def test_p0_all_kept(self) -> None:
        mask = dropout_mask(20, p=0.0)
        assert mask.all()

    def test_p1_none_kept(self) -> None:
        mask = dropout_mask(20, p=1.0)
        assert not mask.any()

    def test_correct_shape(self) -> None:
        mask = dropout_mask(50, p=0.3, seed=0)
        assert mask.shape == (50,)
        assert mask.dtype == torch.bool

    def test_reproducible_with_seed(self) -> None:
        a = dropout_mask(30, p=0.4, seed=7)
        b = dropout_mask(30, p=0.4, seed=7)
        assert torch.equal(a, b)

    def test_invalid_p_raises(self) -> None:
        for bad_p in (-0.1, 1.1):
            try:
                dropout_mask(10, p=bad_p)
                assert False, f"should have raised for p={bad_p}"
            except ValueError:
                pass


class TestExports:
    def test_helpers_exported(self) -> None:
        from lnn.core import (
            apply_input_dropout_to_input,  # noqa: F401
            dropout_mask as dm,  # noqa: F401
            input_dropout as ido,  # noqa: F401
            temporal_dropout as td,  # noqa: F401
        )
        assert callable(dm)
        assert callable(td)
        assert callable(ido)
        assert callable(apply_input_dropout_to_input)


class TestInputDropout:
    """Round 93 (PRD #10-55) — input-side temporal dropout."""

    def test_p0_returns_unchanged(self) -> None:
        t = torch.linspace(0, 1, 10)
        y = torch.sin(2 * 3.14159 * t)
        t_out, y_out = input_dropout(t, y, p=0.0)
        assert torch.equal(t_out, t)
        assert torch.equal(y_out, y)

    def test_p1_returns_all_zeros(self) -> None:
        t = torch.linspace(0, 1, 10)
        y = torch.ones(10) * 5.0
        _, y_out = input_dropout(t, y, p=1.0)
        assert torch.all(y_out == 0.0)

    def test_reproducible_with_seed(self) -> None:
        t = torch.linspace(0, 1, 50)
        y = torch.ones(50)
        _, y_a = input_dropout(t, y, p=0.3, seed=0)
        _, y_b = input_dropout(t, y, p=0.3, seed=0)
        assert torch.equal(y_a, y_b)

    def test_invalid_p_raises(self) -> None:
        t = torch.linspace(0, 1, 10)
        y = torch.ones(10)
        for bad_p in (-0.1, 1.1):
            try:
                input_dropout(t, y, p=bad_p)
                assert False, f"should have raised for p={bad_p}"
            except ValueError:
                pass

    def test_apply_input_dropout_to_input_helper(self) -> None:
        """Convenience wrapper returns only the masked y."""
        t = torch.linspace(0, 1, 8)
        y = torch.ones(8) * 3.0
        out = apply_input_dropout_to_input(t, y, p=0.5, seed=2)
        assert out.shape == y.shape
        # Same as input_dropout(_, _, p=0.5, seed=2)[1].
        _, expected = input_dropout(t, y, p=0.5, seed=2)
        assert torch.equal(out, expected)

    def test_equivalence_to_temporal_dropout(self) -> None:
        """input_dropout and temporal_dropout are functionally identical.

        The distinction is semantic: caller treats masked values as
        missing INPUTS (passed to the model) vs missing TARGETS (in
        the loss). Stateless models behave identically; stateful
        models differ in interpretation, not in the tensor.
        """
        torch.manual_seed(99)
        t = torch.linspace(0, 1, 100)
        y = torch.rand(100)
        _, y_target = temporal_dropout(t, y, p=0.3, seed=5)
        _, y_input = input_dropout(t, y, p=0.3, seed=5)
        assert torch.equal(y_target, y_input)
