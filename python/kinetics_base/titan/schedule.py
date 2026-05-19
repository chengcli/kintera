"""KINETICS-base Titan ``DELTIM`` / stage-based timestep schedule.

This mirrors the schedule that the KB Fortran driver uses when invoked with
the Titan run input (``ions_c6h7+_H2CN.inp-1``).

KB's input read at ``kinetgen1X.F:2077`` takes only 6 fields:
  ``ICYEAR DAY HOUR DELTIM DELTTRM NCYCLE``
The Titan inp-1 file has 7 fields (with ``DELTEM=0`` inserted before
``NCYCLE``), so KB reads ``NCYCLE=0`` and the default ``NCYCLE=2`` kicks
in at line 2114. The on-paper ``NCYCLE=10`` in the inp file is effectively
IGNORED. The actual ``DELT`` growth factor in KB's Titan run is therefore
``10^(1/2) = √10 ≈ 3.16x`` per step.

KB's UPDATEB stage-cap (``TSTEPX`` / ``TELAPSE`` / ``NSTAGE`` tables) is
gated by ``#ifdef __EARTH``; the Titan binary does NOT define ``__EARTH``,
so ``DELT`` just keeps growing by ``ramp_factor`` forever, capped only at
``max_dt``. Observed in the kb_run_50 log: step 50 reaches ``DELT ≈ 1e+9
s`` (~30 years).

Phase 1c refactor (REFACTOR_SCHEMA.html §5): the generic
geometric-growth schedule now lives at :mod:`atm2d.schedule`; this
function is a thin KB-defaults wrapper.
"""

from __future__ import annotations

from ...atm2d.schedule import StageScheduleConfig, stage_schedule


# Default NCYCLE applied by KB when the inp file's 7th field is absent
# (Titan ions_c6h7+_H2CN.inp-1 — see module docstring).
_KB_TITAN_EFFECTIVE_NCYCLE = 2


def kinetics_base_titan_dt_schedule(
    ntime: int,
    *,
    deltim: float = -1.0e-15,
    ncycle: int = _KB_TITAN_EFFECTIVE_NCYCLE,
    max_dt: float = 1.0e+9,
    branch: int = 1,
) -> list[float]:
    """Return KB's per-step ``DELT`` sequence for ``ntime`` steps.

    Parameters
    ----------
    ntime:
        Number of timesteps. Must be positive.
    deltim:
        ``DELTIM`` from the KB run input. Negative values enable the
        ``10^(1/NCYCLE)`` warm-up starting from ``|DELTIM|``. Positive
        values pin ``DELT`` to ``DELTIM`` and never warm up.
    ncycle:
        ``NCYCLE`` from the run input — number of warm-up steps per decade.
    max_dt:
        Hard cap on dt (seconds). Set to ``None`` or ``0`` to disable.
    branch:
        KB branch (1 ``ISTART=0`` initial, 2 continuation, 3 months 3/9).
        Currently unused — the Titan binary ignores branch stage caps
        (``#ifndef __EARTH``).

    Returns
    -------
    List of ``ntime`` floats — target ``DELT`` per step in seconds. The
    adaptive controller in :mod:`atm2d.timestep` is free to subdivide any
    of these.
    """
    if ntime <= 0:
        raise ValueError(f"ntime must be positive, got {ntime}")
    del branch  # Titan binary has no __EARTH stage cap; branch is informational.
    growth_factor = 10.0 ** (1.0 / int(ncycle)) if deltim < 0 else 1.0
    cfg = StageScheduleConfig(
        start_dt=abs(float(deltim)),
        growth_factor=growth_factor,
        max_dt=float(max_dt) if (max_dt is not None and max_dt > 0) else float("inf"),
    )
    return stage_schedule(ntime, cfg)


__all__ = ["kinetics_base_titan_dt_schedule"]
