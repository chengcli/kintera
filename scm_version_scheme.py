"""setuptools_scm version scheme for kintera: guess-next-dev with a release floor.

setuptools_scm derives the version from git tags. A tree without usable git
metadata takes ``[tool.setuptools_scm] fallback_version`` from pyproject.toml,
except on one path: a ``git archive`` whose ``.git_archival.txt`` carries the
commit but no ``describe-name`` (an archive made by a git older than 2.32,
which cannot expand ``%(describe)``, or one taken from a clone with no
reachable release tag) is reported as ``0.0`` and the fallback is never
consulted (#125). A consumer's version floor then refuses the package.

This scheme is ``guess-next-dev`` with ``fallback_version`` applied as a
floor: no tree reports a version below the release recorded in its own
pyproject.toml. A tree that only knows its commit reports
``<fallback_version>+<node>``, which is at least the floor and still
distinguishable from the release itself.

It is referenced from pyproject.toml by object reference
(``version_scheme = "scm_version_scheme:floored_guess_next_dev"``), so it
applies to every reader of that configuration: ``pip install``,
``python -m build`` and ``python -m setuptools_scm``.
"""

from __future__ import annotations

from setuptools_scm.version import ScmVersion, guess_next_dev_version


def floored_guess_next_dev(version: ScmVersion) -> str:
    """``guess-next-dev``, unless the tag is below ``fallback_version``.

    Below the floor the result is the floor itself, with the commit as a
    local segment when the tree knows it and the local scheme would add
    nothing (an exact, clean ``ScmVersion``).
    """
    floor = version.config.fallback_version
    if not floor:
        return guess_next_dev_version(version)
    floor_version = version.config.version_cls(floor)
    if version.tag >= floor_version:
        return guess_next_dev_version(version)
    if version.exact and version.node:
        node = version.format_with("{node}")
        if not node.startswith("g"):  # setuptools_scm marks git nodes with a g
            node = "g" + node
        return f"{floor_version}+{node}"
    return str(floor_version)
