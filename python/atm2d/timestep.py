"""Adaptive timestep controller for implicit chemistry/transport solves.

The controller wraps a user-supplied ``step_fn(state, dt) -> new_concentration``
primitive. For each target timestep it advances by ``dt_target`` seconds, but
subdivides ``dt`` and retries when the candidate solve is rejected.

A solve is rejected when the candidate concentration tensor is non-finite,
contains severely negative values relative to the per-species scale, or grows
beyond an absolute magnitude cap. On rejection the attempted ``dt`` is halved;
on acceptance the next attempt is allowed to grow back up to ``grow_factor``
times the last accepted step, capped at the remaining time.

The controller is intentionally generic. KINETICS-base Titan boundary pinning,
Dirichlet rows, and clamping should be applied inside ``step_fn`` so that the
candidate concentration the controller inspects already reflects them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch


class _StateLike(Protocol):
    concentration: torch.Tensor


StepFn = Callable[[_StateLike, float], torch.Tensor]
AcceptFn = Callable[[torch.Tensor, torch.Tensor], "AcceptDecision"]


@dataclass
class AcceptDecision:
    accepted: bool
    reason: str = ""


@dataclass
class StepRecord:
    dt: float
    action: str
    elapsed: float


@dataclass
class AdvanceResult:
    concentration: torch.Tensor
    accepted_steps: int
    rejected_steps: int
    last_accepted_dt: float
    max_accepted_dt: float
    records: list[StepRecord] = field(default_factory=list)


def default_accept(
    new_conc: torch.Tensor,
    old_conc: torch.Tensor,
    *,
    relative_negative_tol: float = 1e-3,
    species_scale_floor: float = 1.0,
    absolute_magnitude_cap: float = 1e30,
) -> AcceptDecision:
    """Default acceptance check for ``adaptive_advance``.

    Rejects when the candidate state is non-finite, has any value below
    ``-relative_negative_tol * species_scale``, or any value above
    ``absolute_magnitude_cap``. ``species_scale`` is the per-species max of
    ``|old_conc|``, floored at ``species_scale_floor`` to avoid spuriously
    tight tolerances on trace species starting at zero.
    """
    if not torch.isfinite(new_conc).all():
        return AcceptDecision(False, "non_finite")
    species_scale = old_conc.abs().amax(dim=tuple(range(old_conc.dim() - 1)))
    species_scale = species_scale.clamp(min=species_scale_floor)
    neg_threshold = -relative_negative_tol * species_scale
    bcast_shape = (1,) * (new_conc.dim() - 1) + (new_conc.shape[-1],)
    if (new_conc < neg_threshold.view(bcast_shape)).any():
        return AcceptDecision(False, "severe_negative")
    if absolute_magnitude_cap is not None and (
        new_conc.abs() > absolute_magnitude_cap
    ).any():
        return AcceptDecision(False, "magnitude_cap")
    return AcceptDecision(True)


def adaptive_advance(
    state: _StateLike,
    dt_target: float,
    step_fn: StepFn,
    *,
    accept_fn: AcceptFn | None = None,
    max_subdivisions: int = 16,
    grow_factor: float = 2.0,
    shrink_factor: float = 0.5,
    record_trace: bool = False,
    initial_attempt: float | None = None,
) -> AdvanceResult:
    """Advance ``state.concentration`` by ``dt_target`` with rejection/retry.

    The starting concentration is read from ``state.concentration``. On each
    accepted substep the controller writes the candidate back to
    ``state.concentration``. On rejection the state is left untouched and
    ``dt`` is multiplied by ``shrink_factor`` for the next attempt.

    Parameters
    ----------
    dt_target:
        Total time to advance, in the same units as ``dt`` passed to
        ``step_fn``. Must be positive.
    step_fn:
        Callable ``step_fn(state, dt) -> new_concentration``. Must not mutate
        ``state``. Any boundary pinning or clamping the caller relies on must
        already be applied to the returned tensor.
    accept_fn:
        Acceptance predicate; defaults to ``default_accept``.
    max_subdivisions:
        Maximum number of dt halvings allowed below ``dt_target`` before the
        controller gives up and raises.
    record_trace:
        When true, populate ``AdvanceResult.records`` with one entry per
        attempted substep. Off by default to keep cost low.
    initial_attempt:
        Optional starting ``dt`` for the first substep. Useful when the caller
        knows from a previous ``adaptive_advance`` call that the network can
        only sustain small steps; passing the previous ``last_accepted_dt``
        avoids re-discovering the safe range via repeated rejection from
        ``dt_target``. Clamped to ``dt_target``.
    """
    if dt_target <= 0.0:
        raise ValueError(f"dt_target must be positive, got {dt_target}")
    if accept_fn is None:
        accept_fn = default_accept

    remaining = float(dt_target)
    if initial_attempt is not None and initial_attempt > 0.0:
        next_attempt = min(float(initial_attempt), float(dt_target))
    else:
        next_attempt = float(dt_target)
    floor_dt = next_attempt * (shrink_factor ** max_subdivisions)
    last_accepted = 0.0
    max_accepted = 0.0
    min_rejected = float("inf")
    accepted_steps = 0
    rejected_steps = 0
    records: list[StepRecord] = []
    elapsed = 0.0

    while remaining > 0.0:
        attempt = min(next_attempt, remaining)
        if attempt >= min_rejected:
            attempt = min_rejected * shrink_factor
        if attempt < floor_dt:
            raise RuntimeError(
                "adaptive_advance: dt collapsed below floor "
                f"({attempt:.3e} < {floor_dt:.3e}); remaining={remaining:.3e}, "
                f"rejected_so_far={rejected_steps}"
            )

        new_conc = step_fn(state, attempt)
        decision = accept_fn(new_conc, state.concentration)

        if decision.accepted:
            state.concentration = new_conc
            remaining -= attempt
            elapsed += attempt
            last_accepted = attempt
            if attempt > max_accepted:
                max_accepted = attempt
            accepted_steps += 1
            next_attempt = attempt * grow_factor
            if record_trace:
                records.append(StepRecord(dt=attempt, action="accept", elapsed=elapsed))
        else:
            rejected_steps += 1
            if attempt < min_rejected:
                min_rejected = attempt
            next_attempt = attempt * shrink_factor
            if record_trace:
                records.append(
                    StepRecord(
                        dt=attempt,
                        action=f"reject_{decision.reason}",
                        elapsed=elapsed,
                    )
                )

    return AdvanceResult(
        concentration=state.concentration,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        last_accepted_dt=last_accepted,
        max_accepted_dt=max_accepted,
        records=records,
    )
