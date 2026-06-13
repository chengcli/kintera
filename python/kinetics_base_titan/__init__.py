"""Backward-compat shim for the old ``kintera.kinetics_base_titan`` path.

Phase 2 of the refactor moved this package to
:mod:`kintera.kinetics_base.titan`. This shim keeps ``import kintera``
working for code that still uses the old name (notably the ``kintera``
top-level ``from .kinetics_base_titan import *`` line in
``python/__init__.py``).

New code should import from :mod:`kintera.kinetics_base` or
:mod:`kintera.kinetics_base.titan` directly.
"""

from kintera.kinetics_base.titan import *  # noqa: F401,F403
from kintera.kinetics_base.titan import __all__ as _all
from kintera.kinetics_base import titan  # noqa: F401

__all__ = list(_all)
