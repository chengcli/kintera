"""The release floor in pyproject.toml bounds every kintera tree from below.

setuptools_scm reads the version from git tags; [tool.setuptools_scm]
fallback_version is what a tree without git metadata reports. One path
bypassed it: a `git archive` whose .git_archival.txt carries the commit but
no describe-name (an archive made by a git older than 2.32, which cannot
expand %(describe), or one taken from a clone with no reachable release tag)
was reported as 0.0 (#125). scm_version_scheme.py, referenced from
pyproject.toml, applies fallback_version as a floor on that path too.

The cases export HEAD with `git archive`, rewrite .git_archival.txt into each
form an old or tagless git leaves behind, and ask setuptools_scm for the
version of that tree, as pip does at install time.
"""

import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("setuptools_scm")
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DESCRIBE_UNEXPANDED = "%(describe:tags=true,match=v[0-9]*)"


def floor():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return Version(tomllib.load(f)["tool"]["setuptools_scm"]["fallback_version"])


def git(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        pytest.skip(f"git {args[0]} failed: {out.stderr.decode().strip()}")
    return out.stdout


@pytest.fixture
def exported(tmp_path):
    """HEAD extracted from `git archive`, with its commit hash."""
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=git("archive", "HEAD"),
                   check=True)
    return tmp_path, git("rev-parse", "HEAD").decode().strip()


def scm_version(tree):
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tree.parent))
    out = subprocess.run([sys.executable, "-m", "setuptools_scm"], cwd=tree,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return Version(out.stdout.strip().splitlines()[-1])


def write_archival(tree, **fields):
    text = "".join(f"{key.replace('_', '-')}: {value}\n" for key, value in fields.items())
    (tree / ".git_archival.txt").write_text(text)


def test_floor_is_not_ahead_of_the_newest_release():
    """fallback_version is a release that exists in this history, so a tree
    never claims a version newer than the tags it was cut from."""
    newest = Version(git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*")
                     .decode().strip().lstrip("v"))
    assert floor() <= newest


def test_archive_by_an_old_git_reports_at_least_the_floor(exported):
    """git < 2.32 expands %H but not %(describe) in .git_archival.txt."""
    tree, head = exported
    write_archival(tree, node=head, node_date="%cI", describe_name=DESCRIBE_UNEXPANDED,
                   ref_names="  (HEAD, main)")
    version = scm_version(tree)
    assert version >= floor()
    assert version.public == str(floor())
    assert version.local.startswith("g" + head[:7]), version


def test_archive_of_a_release_by_an_old_git_reports_that_release(exported):
    """At a release tag an old git still records the tag in ref-names (%d)."""
    tree, head = exported
    write_archival(tree, node=head, node_date="%cI", describe_name=DESCRIBE_UNEXPANDED,
                   ref_names=f"  (HEAD, tag: v{floor()}, main)")
    assert scm_version(tree) == floor()


def test_archive_from_a_clone_without_tags_reports_at_least_the_floor(exported):
    """A recent git leaves describe-name empty when no tag is reachable."""
    tree, head = exported
    write_archival(tree, node=head, node_date="2026-01-01T00:00:00+00:00",
                   describe_name="", ref_names="HEAD -> main")
    version = scm_version(tree)
    assert version >= floor()
    assert version.local.startswith("g" + head[:7]), version


def test_tree_without_git_metadata_reports_the_floor(exported):
    """A plain copy of the sources: no .git, no archival record."""
    tree, _ = exported
    (tree / ".git_archival.txt").unlink()
    assert scm_version(tree) == floor()


OLDER_RELEASE = "0.1.0"


def git_in(tree, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               GIT_CEILING_DIRECTORIES=str(tree.parent))
    subprocess.run(["git", *args], cwd=tree, check=True, capture_output=True, env=env)


def test_clone_descended_from_an_older_release_reports_its_own_version(exported):
    """A clone whose newest reachable tag is below the floor (a branch cut
    before that release) is versioned from its own tag, not labelled as the
    floor release: the floor is only for a tree with no usable tag."""
    tree, _ = exported
    git_in(tree, "init", "-q")
    git_in(tree, "add", "-A")
    git_in(tree, "commit", "-qm", "an older release")
    git_in(tree, "tag", f"v{OLDER_RELEASE}")
    assert scm_version(tree) == Version(OLDER_RELEASE)
    for message in ("one", "two"):
        git_in(tree, "commit", "-q", "--allow-empty", "-m", message)
    version = scm_version(tree)
    assert version.public == "0.1.1.dev2", version
    assert version < floor()


def test_this_checkout_is_versioned_from_its_own_tags():
    """The floor never replaces the version of a checkout with a usable tag."""
    describe = git("describe", "--tags", "--long", "--match", "v[0-9]*").decode().strip()
    tag, distance, _ = describe.rsplit("-", 2)
    tag, distance = Version(tag.lstrip("v")), int(distance)
    version = scm_version(ROOT)
    if distance == 0 and version.dev is None:
        assert version.public == str(tag)
    else:
        assert version.release == (tag.major, tag.minor, tag.micro + 1), version
        assert version.dev == distance, version
