"""KINETICS-base Titan ``DELTIM`` / stage-based timestep schedule.

This mirrors the schedule that the KB Fortran driver uses when invoked with the
Titan run input (``ions_c6h7+_H2CN.inp-1``). It replaces the bogus
``1e-15 × √10`` placeholder the kintera diagnostics were using.

Two ingredients combine in KB to produce the actual ``DELT`` sequence:

1. **Stage table** (``TSTEPX`` / ``TELAPSE`` / ``NSTAGE``) — once the chemistry
   has settled, KB advances at a fixed dt per stage and moves to the next
   stage after a fixed amount of *physical* time elapses in that stage.
2. **Negative-DELTIM exponential warm-up** — when the run input gives
   ``DELTIM < 0`` (the Titan default is ``-1e-15`` with ``NCYCLE=10``), KB
   ramps ``DELT`` from ``|DELTIM|`` by a factor of ``10^(1/NCYCLE)`` per step,
   capped by the current stage's ``TSTEPX``. The first ``DELT`` reaches the
   60 s stage-1 size at about step 167; before that the chemistry is
   essentially exploring its initial transients.

Source references (under ``diagnostics/KINETICS-base-compare/src/KINETGENX/``):

- ``kinetgen2X.F`` lines 14911–14917: ``TSTEPX``, ``TELAPSEX``, ``NSTAGEX``
  tables.
- ``kinetgen2X.F`` lines 14326–14335: warm-up logic (``DELTIN``, negative
  ``DELTIM``, ``10^(1/NCYCLE)`` ramp).
- ``kinetgen2X.F`` lines 14941–14961: branch selection and stage advance.
- ``kinetgen1X.F`` lines 98–123: the ``CHIEF`` outer timestep loop.

The ``DELT`` reported by the schedule here is the *target* dt for that step.
kintera's adaptive controller is free to subdivide that target if its
acceptance check rejects the proposed solve; KB does the same internally via
``RETRY`` (subdivides by ``2**(NTRYS-1)``).
"""

from __future__ import annotations

from dataclasses import dataclass


# DELT values per stage (seconds), per branch. Source: TSTEPX in kinetgen2X.F:14911-14913.
_TSTEP_TABLE: dict[int, tuple[float, ...]] = {
    1: (60.0, 600.0, 3600.0, 7200.0, 10800.0),
    2: (600.0, 3600.0, 7200.0, 10800.0),
    3: (60.0, 600.0, 3600.0, 5400.0),
}

# Elapsed time per stage (hours -> seconds), per branch. Source: TELAPSEX in kinetgen2X.F:14914-14916.
_TELAPSE_HOURS_TABLE: dict[int, tuple[float, ...]] = {
    1: (1.0, 1.0, 6.0, 16.0, 1.0e30),
    2: (1.0, 7.0, 16.0, 1.0e30),
    3: (1.0, 1.0, 22.0, 1.0e30),
}

# Tolerance on the ESTAGE >= TELAPSE check (EPSIL in kinetgen2X.F:14918).
_EPSIL = 1.0e-5


@dataclass(frozen=True)
class _ScheduleConfig:
    deltim: float
    ncycle: int
    branch: int
    tstep: tuple[float, ...]
    telapse_sec: tuple[float, ...]


def _config(branch: int, deltim: float, ncycle: int) -> _ScheduleConfig:
    if branch not in _TSTEP_TABLE:
        raise ValueError(f"unknown KB branch {branch}; valid: 1/2/3")
    tstep = _TSTEP_TABLE[branch]
    telapse_sec = tuple(h * 3600.0 for h in _TELAPSE_HOURS_TABLE[branch])
    return _ScheduleConfig(
        deltim=float(deltim),
        ncycle=int(ncycle),
        branch=int(branch),
        tstep=tstep,
        telapse_sec=telapse_sec,
    )


# KB's input read at kinetgen1X.F:2077 takes only 6 fields:
#   ICYEAR, DAY, HOUR, DELTIM, DELTTRM, NCYCLE
# The Titan inp-1 file has 7 fields (with DELTEM=0 inserted before NCYCLE),
# so KB reads NCYCLE=0 and the default NCYCLE=2 kicks in at line 2114.
# The on-paper NCYCLE=10 in the inp file is effectively IGNORED. The actual
# DELT growth factor in KB's Titan run is therefore 10^(1/2) = √10 ≈ 3.16x
# per step, not the 10^(1/10) ≈ 1.26x our earlier reading suggested.
_KB_TITAN_EFFECTIVE_NCYCLE = 2


def kinetics_base_titan_dt_schedule(
    ntime: int,
    *,
    deltim: float = -1.0e-15,
    ncycle: int = _KB_TITAN_EFFECTIVE_NCYCLE,
    max_dt: float = 1.0e+10,
    branch: int = 1,
) -> list[float]:
    """Return KB's per-step ``DELT`` sequence for ``ntime`` steps.

    Parameters
    ----------
    ntime:
        Number of timesteps to generate. Must be positive.
    deltim:
        The ``DELTIM`` field from the KB run input. Negative values enable the
        ``10^(1/NCYCLE)`` exponential warm-up starting from ``|DELTIM|``.
        Positive values pin ``DELT`` to ``DELTIM`` and never warm up.
        Defaults to ``-1.0e-15`` (Titan ``ions_c6h7+_H2CN.inp-1``).
    ncycle:
        ``NCYCLE`` field from the run input — number of warm-up steps per
        decade. Default 10 matches the Titan input.
    branch:
        KB branch (``1`` for ``ISTART=0``, ``2`` for continuation runs,
        ``3`` for months 3/9). Default 1 matches the Titan diagnostic, which
        sets ``ISTART=0`` via ``compare_gch4.py`` / ``no_grain_stability.py``.

    Returns
    -------
    A list of ``ntime`` positive floats — the target ``DELT`` for each KB
    timestep, in seconds. The adaptive controller in ``atm2d.timestep`` is
    free to subdivide any of these targets if the implicit solve is rejected.
    """
    if ntime <= 0:
        raise ValueError(f"ntime must be positive, got {ntime}")
    cfg = _config(branch=branch, deltim=deltim, ncycle=ncycle)

    sequence: list[float] = []
    estage_sec = 0.0
    istage = 0  # zero-based index into cfg.tstep / cfg.telapse_sec
    delt_prev = 0.0

    ramp_factor = 10.0 ** (1.0 / cfg.ncycle) if cfg.deltim < 0 else 1.0
    warmup_active = cfg.deltim < 0

    # The stage-based cap in KB's UPDATEB is gated by ``#ifdef __EARTH``; the
    # Titan binary does NOT define __EARTH so DELT just keeps growing by
    # ramp_factor forever, with no cap. Observed in the kb_run_50 log:
    # step 50 reaches DELT ≈ 1e+9 s (~30 years).
    del istage  # noqa - unused for Titan (no __EARTH stage cap)
    del estage_sec  # noqa - unused for Titan

    for step_idx in range(ntime):
        if warmup_active:
            if step_idx == 0:
                delt = abs(cfg.deltim)
            else:
                delt = delt_prev * ramp_factor
        else:
            delt = abs(cfg.deltim)

        # Cap to avoid astronomically large dt that pushes Newton into
        # non-physical fixed points. KB output saturates by ~NT=50 (where dt
        # naturally reaches ~3e+9 s ≈ 100 years), so any dt beyond ~1e+10 s
        # (3 kyr) is purely numerical and only causes mass-conservation
        # violations.
        if max_dt is not None and delt > max_dt:
            delt = max_dt

        sequence.append(delt)
        delt_prev = delt

    return sequence


__all__ = ["kinetics_base_titan_dt_schedule"]
