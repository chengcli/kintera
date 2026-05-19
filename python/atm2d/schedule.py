"""Generic implicit-marching dt schedule.

For backward-Euler / implicit-Newton drivers it is common to start at a
tiny dt, grow it geometrically until the chemistry has settled, and cap
at some maximum to avoid Newton landing on non-physical fixed points.
This module exposes that pattern as a planet-agnostic schedule; the
adaptive controller in :mod:`atm2d.timestep` is free to subdivide any
proposed dt.

KB Titan defaults (NCYCLE=2 → growth=√10, start=1e-15 s, cap 1e+9 s)
live in :mod:`kinetics_base_titan.schedule`, which is now a thin
wrapper around :func:`stage_schedule` here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageScheduleConfig:
    """Parameters for a geometric-growth dt schedule.

    Parameters
    ----------
    start_dt:
        First step's dt, in seconds. Typical: ``1e-15`` for stiff
        chemistry initial transients (= KB ``|DELTIM|``).
    growth_factor:
        Multiplicative growth per step. ``10.0 ** (1.0 / ncycle)`` in KB
        nomenclature: ``ncycle=2`` gives ``√10 ≈ 3.16``, ``ncycle=10``
        gives ``≈1.26``.
    max_dt:
        Hard cap on dt; once the geometric growth exceeds this, all
        subsequent steps stay at ``max_dt``.
    """

    start_dt: float
    growth_factor: float
    max_dt: float


def stage_schedule(ntime: int, cfg: StageScheduleConfig) -> list[float]:
    """Return ``ntime`` target dt values, geometrically growing from
    ``cfg.start_dt`` by ``cfg.growth_factor`` per step, clipped at
    ``cfg.max_dt``.

    Step 0 always returns ``min(start_dt, max_dt)``. Step ``k > 0``
    returns ``min(start_dt * growth_factor ** k, max_dt)`` — equivalently,
    propagate ``dt *= growth_factor`` after emitting each value, with a
    cap applied at emission time.
    """
    if ntime <= 0:
        raise ValueError(f"ntime must be positive, got {ntime}")
    sequence: list[float] = []
    dt = cfg.start_dt
    for _ in range(ntime):
        sequence.append(min(dt, cfg.max_dt))
        dt = dt * cfg.growth_factor
    return sequence


__all__ = ["StageScheduleConfig", "stage_schedule"]
