"""Unit tests for the adaptive timestep controller."""

from __future__ import annotations

from dataclasses import dataclass

import math

import pytest
import torch

from kintera.atm2d.timestep import (
    AcceptDecision,
    adaptive_advance,
    default_accept,
)


@dataclass
class _ToyState:
    concentration: torch.Tensor


def _make_state(values):
    tensor = torch.tensor(values, dtype=torch.float64).view(1, 1, -1)
    return _ToyState(concentration=tensor)


def test_accepts_single_step_when_step_fn_is_clean():
    state = _make_state([1.0, 2.0, 3.0])

    def step_fn(s, dt):
        return s.concentration * math.exp(-0.1 * dt)

    result = adaptive_advance(state, dt_target=1.0, step_fn=step_fn, record_trace=True)
    assert result.accepted_steps == 1
    assert result.rejected_steps == 0
    assert result.last_accepted_dt == pytest.approx(1.0)
    torch.testing.assert_close(
        state.concentration,
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).view(1, 1, -1) * math.exp(-0.1),
    )


def test_subdivides_when_large_dt_produces_nan():
    """step_fn returns NaN for dt > 0.2; controller should subdivide and converge."""
    state = _make_state([1.0])

    def step_fn(s, dt):
        if dt > 0.2:
            return torch.full_like(s.concentration, float("nan"))
        return s.concentration * (1.0 - 0.05 * dt)

    result = adaptive_advance(state, dt_target=1.0, step_fn=step_fn, record_trace=True)
    assert result.rejected_steps >= 1
    assert result.accepted_steps >= 1
    # Total elapsed time should match dt_target
    elapsed = sum(rec.dt for rec in result.records if rec.action == "accept")
    assert elapsed == pytest.approx(1.0, rel=1e-9)
    # Final concentration should be finite and < 1
    assert torch.isfinite(state.concentration).all()
    assert state.concentration.item() < 1.0


def test_subdivides_when_large_dt_produces_severe_negative():
    """step_fn returns large negatives for dt > 0.3; controller should reject."""
    state = _make_state([100.0])

    def step_fn(s, dt):
        if dt > 0.3:
            return torch.full_like(s.concentration, -50.0)
        return s.concentration - 0.1 * dt

    result = adaptive_advance(state, dt_target=1.0, step_fn=step_fn, record_trace=True)
    assert result.rejected_steps >= 1
    assert torch.isfinite(state.concentration).all()
    # Verify the rejection reason is severe_negative
    reject_reasons = {rec.action for rec in result.records if "reject" in rec.action}
    assert "reject_severe_negative" in reject_reasons


def test_raises_when_dt_collapses_below_floor():
    """A step_fn that always fails should hit the subdivision floor and raise."""
    state = _make_state([1.0])

    def step_fn(s, dt):
        return torch.full_like(s.concentration, float("nan"))

    with pytest.raises(RuntimeError, match="dt collapsed below floor"):
        adaptive_advance(state, dt_target=1.0, step_fn=step_fn, max_subdivisions=3)


def test_does_not_retry_rejected_dt_within_call():
    """Once an attempt at dt=X is rejected within an adaptive_advance call,
    the controller should not try X (or larger) again — it caps subsequent
    attempts strictly below the smallest rejected dt. This avoids wasting
    Newton calls re-discovering the safe range every substep."""
    state = _make_state([1.0])
    attempted = []

    def step_fn(s, dt):
        attempted.append(dt)
        if dt > 0.4:
            return torch.full_like(s.concentration, float("nan"))
        return s.concentration * 0.999

    adaptive_advance(state, dt_target=1.6, step_fn=step_fn, record_trace=True, grow_factor=2.0)
    # Expected sequence: 1.6 (reject), 0.8 (reject), 0.4 (accept), then
    # subsequent attempts capped below 0.8.
    assert attempted[0] == pytest.approx(1.6)
    assert attempted[1] == pytest.approx(0.8)
    # After the rejection at 0.8, every later attempt must be < 0.8.
    later = attempted[2:]
    assert later, "expected at least one accepted attempt"
    assert max(later) < 0.8, f"later attempts should stay below rejected 0.8: {later}"


def test_default_accept_non_finite():
    old = torch.tensor([1.0, 2.0]).view(1, 1, -1)
    new_nan = torch.tensor([1.0, float("nan")]).view(1, 1, -1)
    new_inf = torch.tensor([float("inf"), 2.0]).view(1, 1, -1)
    assert not default_accept(new_nan, old).accepted
    assert default_accept(new_nan, old).reason == "non_finite"
    assert not default_accept(new_inf, old).accepted


def test_default_accept_tolerates_small_negatives():
    """Tiny negative values from solver noise should be accepted."""
    old = torch.tensor([1e15, 1e10]).view(1, 1, -1)
    new = torch.tensor([1e15, -1.0]).view(1, 1, -1)  # -1 is tiny vs species_scale_floor=1
    decision = default_accept(new, old)
    assert decision.accepted, f"expected accept, got reject: {decision.reason}"


def test_default_accept_rejects_severe_negative():
    old = torch.tensor([1e15, 1e10]).view(1, 1, -1)
    new = torch.tensor([1e15, -1e8]).view(1, 1, -1)  # -1e8 > 1e-3 * 1e10
    decision = default_accept(new, old)
    assert not decision.accepted
    assert decision.reason == "severe_negative"


def test_default_accept_rejects_magnitude_cap():
    old = torch.tensor([1.0, 1.0]).view(1, 1, -1)
    new = torch.tensor([1.0, 1e35]).view(1, 1, -1)
    decision = default_accept(new, old)
    assert not decision.accepted
    assert decision.reason == "magnitude_cap"


def test_state_not_mutated_on_rejection():
    """If all attempts fail, state.concentration should remain at its start value."""
    state = _make_state([1.0, 2.0, 3.0])
    initial = state.concentration.clone()

    def step_fn(s, dt):
        return torch.full_like(s.concentration, float("nan"))

    with pytest.raises(RuntimeError):
        adaptive_advance(state, dt_target=1.0, step_fn=step_fn, max_subdivisions=4)
    torch.testing.assert_close(state.concentration, initial)
