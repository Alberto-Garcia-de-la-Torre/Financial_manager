"""The commit guardrails in `tools/pre-commit-config.yaml`.

The rule these tests defend is the one that cannot be undone: a Parquet file
committed today is in every clone forever, and `git rm` tomorrow does not get
it back out. `.gitignore` hides `data/` but does not defend it — `git add -f`
walks straight past it — so the hook is what actually says no.

Asserting on the regex alone would test a string, not a guarantee. Each test
below builds a throwaway git repository under `tmp_path`, installs this
project's real hook configuration into it, and attempts a real commit. The
repository under test is disposable; this repository is never touched.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from finmgr.config import PROJECT_ROOT

HOOK_CONFIG = PROJECT_ROOT / "tools" / "pre-commit-config.yaml"

# A minimal Parquet file: the magic bytes at both ends are what makes git treat
# it as binary. Nothing here needs to parse it, only to try to commit it.
PARQUET_BYTES = b"PAR1" + b"\x00" * 64 + b"PAR1"


@dataclass(frozen=True)
class CommitAttempt:
    """The outcome of one `git commit`, with the hook's own words kept."""

    accepted: bool
    # git sends hook output to stderr and its own summary to stdout; which is
    # which is not worth asserting on, so the tests read them as one stream.
    output: str


@dataclass(frozen=True)
class HookRepo:
    """A disposable git repository with this project's hooks installed."""

    path: Path
    env: dict[str, str]

    def run(self, *command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command, cwd=self.path, env=self.env, capture_output=True, text=True, check=False
        )

    def write(self, relative: str, content: str | bytes) -> None:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")

    def commit(self, *paths: str, force: bool = False) -> CommitAttempt:
        """Stage `paths` and attempt a commit, reporting what git did."""
        staged = self.run("git", "add", *(("-f",) if force else ()), *paths)
        assert staged.returncode == 0, staged.stderr
        result = self.run("git", "commit", "-m", "attempt")
        return CommitAttempt(accepted=result.returncode == 0, output=result.stdout + result.stderr)

    @property
    def commits(self) -> int:
        result = self.run("git", "rev-list", "--count", "HEAD")
        return int(result.stdout.strip()) if result.returncode == 0 else 0


@pytest.fixture
def hook_repo(tmp_path: Path) -> HookRepo:
    repo = HookRepo(
        path=tmp_path / "clone",
        # The developer's own git config must not reach in here: a global
        # core.hooksPath or commit.gpgsign would make these tests lie, in one
        # direction or the other. PRE_COMMIT_HOME keeps the cache disposable too.
        env={
            "PATH": f"{PROJECT_ROOT / '.venv' / 'bin'}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "PRE_COMMIT_HOME": str(tmp_path / "pre-commit-cache"),
        },
    )
    (tmp_path / "home").mkdir()
    (repo.path / ".venv" / "bin").mkdir(parents=True)

    # The ruff hooks run `.venv/bin/ruff`, resolved against the repository root.
    # Symlinking the real pinned binary lets the fixture exercise the actual
    # configuration file instead of a trimmed-down copy of it.
    (repo.path / ".venv" / "bin" / "ruff").symlink_to(PROJECT_ROOT / ".venv" / "bin" / "ruff")

    # Same shape as the real repository: data/ is ignored, so staging anything
    # out of it takes the deliberate `git add -f` the acceptance criterion means.
    repo.write(".gitignore", "/data/\n.venv/\n")
    repo.write("tools/pre-commit-config.yaml", HOOK_CONFIG.read_text(encoding="utf-8"))

    repo.run("git", "init", "--initial-branch=main")
    repo.run("git", "config", "user.email", "test@example.invalid")
    repo.run("git", "config", "user.name", "Hook Test")

    installed = repo.run(
        sys.executable, "-m", "pre_commit", "install", "--config", "tools/pre-commit-config.yaml"
    )
    assert installed.returncode == 0, installed.stderr
    assert (repo.path / ".git" / "hooks" / "pre-commit").exists()
    return repo


# ---------------------------------------------------------------------------
# The acceptance criterion: a deliberate attempt to commit a Parquet file
# ---------------------------------------------------------------------------


def test_a_parquet_file_under_data_cannot_be_committed(hook_repo: HookRepo) -> None:
    """`git add -f data/bars/... && git commit` must not produce a commit."""
    hook_repo.write("data/bars/ticker=AAPL/part.parquet", PARQUET_BYTES)

    result = hook_repo.commit("data/bars/ticker=AAPL/part.parquet", force=True)

    assert not result.accepted
    assert "data/bars/ticker=AAPL/part.parquet" in result.output
    assert "Market data does not belong in git" in result.output
    assert hook_repo.commits == 0


def test_a_parquet_file_anywhere_else_is_refused_too(hook_repo: HookRepo) -> None:
    """Moving it out of data/ is not a workaround; the extension is enough."""
    hook_repo.write("panel.parquet", PARQUET_BYTES)

    result = hook_repo.commit("panel.parquet")

    assert not result.accepted
    assert "panel.parquet" in result.output
    assert hook_repo.commits == 0


def test_any_file_under_data_is_refused_not_just_parquet(hook_repo: HookRepo) -> None:
    """The rule is about the directory, not about one file format."""
    hook_repo.write("data/notes.txt", "scratch notes\n")

    result = hook_repo.commit("data/notes.txt", force=True)

    assert not result.accepted
    assert hook_repo.commits == 0


# ---------------------------------------------------------------------------
# ... and the other half: ordinary work still commits
# ---------------------------------------------------------------------------


def test_source_code_commits_normally(hook_repo: HookRepo) -> None:
    """Including `finmgr/data/`, which is the storage layer, not the data."""
    hook_repo.write("finmgr/data/store.py", '"""Bar storage."""\n\nBARS = "bars"\n')
    hook_repo.write("config/universe.csv", "ticker,name\nAAPL,Apple\n")

    result = hook_repo.commit("finmgr/data/store.py", "config/universe.csv")

    assert result.accepted, result.output
    assert hook_repo.commits == 1


def test_badly_formatted_python_is_rejected_by_ruff(hook_repo: HookRepo) -> None:
    """The formatter rewrites the file, which fails the commit as intended."""
    hook_repo.write("ugly.py", "x  =  1\n")

    result = hook_repo.commit("ugly.py")

    assert not result.accepted
    assert hook_repo.commits == 0
    assert (hook_repo.path / "ugly.py").read_text(encoding="utf-8") == "x = 1\n"


# ---------------------------------------------------------------------------
# The configuration itself
# ---------------------------------------------------------------------------


def test_the_config_declares_the_three_hooks() -> None:
    config = yaml.safe_load(HOOK_CONFIG.read_text(encoding="utf-8"))
    ids = [hook["id"] for repo in config["repos"] for hook in repo["hooks"]]

    assert ids == ["no-data-in-git", "ruff-check", "ruff-format"]
