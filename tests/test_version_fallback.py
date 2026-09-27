"""A kintera tree without git metadata must not report a version older than
its newest release.

setuptools_scm uses [tool.setuptools_scm] fallback_version whenever it cannot
read git (a `git archive` tarball, or a git too old for setuptools_scm). That
version becomes the installed metadata: kintera.__version__, and what a
consumer's find_package(Kintera X.Y) compares. The case exports HEAD with
`git archive`, as a release tarball is made, and asks setuptools_scm for the
version of that tree.
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_archive_version_is_not_older_than_the_newest_release(tmp_path):
    pytest.importorskip("setuptools_scm")
    from packaging.version import Version

    # cwd rather than `git -C`, which needs git 1.8.5: the case must run, not
    # skip, with the old git of #125.
    tag = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
        cwd=ROOT, capture_output=True, text=True)
    if tag.returncode != 0:
        pytest.skip("no release tag reachable (shallow clone without tags)")
    newest = Version(tag.stdout.strip().lstrip("v"))

    archive = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT,
                             capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=archive.stdout,
                   check=True)
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp_path.parent))
    out = subprocess.run([sys.executable, "-m", "setuptools_scm"],
                         cwd=tmp_path, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    version = Version(out.stdout.strip().splitlines()[-1])
    assert version >= newest, (
        f"a tree without git metadata reports {version}, older than the "
        f"release {newest} it was exported from")
