"""kinetics_base — KB Fortran network adapter for kintera.

Phase 2 of the L1/L2 refactor (REFACTOR_SCHEMA.html). All Titan-specific
content currently lives in :mod:`kintera.kinetics_base.titan`. Future
phases (3+) will split out:

  * :mod:`kintera.kinetics_base.io`       — pun / special / bc_save / atm parsers
  * :mod:`kintera.kinetics_base.physics`  — generic radiation / electron impact / photochemistry helpers
  * :mod:`kintera.kinetics_base.titan`    — Titan-only physics (cold trap, grain sources)
  * ``kinetics_base.adapter``             — KB source → atm2d adapter
  * ``kinetics_base.schedule_defaults``   — KB run-input defaults

For now the public surface lives entirely under ``.titan``, and the
legacy ``kintera.kinetics_base_titan`` import path is preserved as a
backward-compat shim.
"""

from .titan import *  # noqa: F401,F403
from .titan import __all__ as _titan_all

__all__ = list(_titan_all)
