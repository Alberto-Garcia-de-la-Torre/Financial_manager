#!/usr/bin/env python3
"""Sixty Sessions - run one roadmap step a day, unattended.

Reads ROADMAP.md, works out which step is next, hands that step to Claude
Code in headless mode, verifies the result, then commits and pushes it.

  roadmap_runner.py status              what's done, what's next
  roadmap_runner.py run                 do the next step
  roadmap_runner.py run --day 7         do a specific step
  roadmap_runner.py run --dry-run       print the prompt, change nothing
  roadmap_runner.py preflight           check the machine is set up
  roadmap_runner.py resync              rebuild state from git history

State lives outside the repo (~/.local/state/finmgr-roadmap/state.json) so
it never collides with a branch or a merge. `resync` rebuilds it from the
commit trailers if it is ever lost.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "finmgr-roadmap"
STATE_FILE = STATE_DIR / "state.json"
LOCK_FILE = STATE_DIR / "runner.lock"
LOG_DIR = STATE_DIR / "logs"

TRAILER = "Roadmap-Day:"  # how a completed day is recorded in git history

DEFAULTS = {
    "REPO_DIR": str(HERE.parent),
    "ROADMAP_FILE": "ROADMAP.md",
    "RUN_TIME": "19:30",
    "MODEL": "opus",
    "STEP_TIMEOUT": "3600",
    "ALLOWED_TOOLS": "Read,Write,Edit,Glob,Grep,TodoWrite,Bash",
    "DENIED_TOOLS": ("Bash(git commit:*),Bash(git push:*),Bash(git checkout:*),"
                     "Bash(git reset:*),Bash(git rebase:*)"),
    "AGENT_CMD": ("claude -p {prompt} --output-format json --model {model} "
                  "--permission-mode acceptEdits --permission-prompts none "
                  "--allowedTools {tools} --disallowedTools {denied}"),
    "BASE_BRANCH": "main",
    "BRANCH_MODE": "branch",
    "PUSH": "1",
    "AUTO_MERGE": "0",
    "OPEN_PR": "0",
    "NOTIFY": "1",
    "ARTIFACT_URL": "",
}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_config() -> dict:
    cfg = dict(DEFAULTS)
    path = HERE / "config.env"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            cfg[key.strip()] = val.strip().strip('"').strip("'")
    # environment wins, so the systemd unit or a one-off run can override
    for key in list(cfg):
        if key in os.environ:
            cfg[key] = os.environ[key]
    return cfg


def flag(cfg: dict, key: str) -> bool:
    return str(cfg.get(key, "0")).lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# roadmap parsing
# --------------------------------------------------------------------------

RE_PHASE = re.compile(r"^##\s+Phase\s+(\d+)\s+[—-]\s+(.+?)\s*$")
RE_DAY = re.compile(r"^###\s+Day\s+(\d+)\s+[—-]\s+(.+?)\s*$")
RE_DONE = re.compile(r"^\*\*Done when:\*\*\s*(.+?)\s*$")
RE_DATE = re.compile(r"^\*(.+?)\*\s*$")


class Step:
    def __init__(self, n, title, phase_n, phase_title, phase_goal, when, body, check):
        self.n = n
        self.title = title
        self.phase_n = phase_n
        self.phase_title = phase_title
        self.phase_goal = phase_goal
        self.when = when
        self.body = body
        self.check = check

    @property
    def slug(self) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")
        return s[:40].rstrip("-")

    @property
    def branch(self) -> str:
        return f"roadmap/day-{self.n:02d}-{self.slug}"


def parse_roadmap(path: Path) -> list:
    if not path.exists():
        die(f"roadmap not found: {path}")
    steps = []
    phase_n = phase_title = phase_goal = None
    cur = None          # day being accumulated
    body = []
    seen_date = False
    pending_goal = False

    def flush():
        nonlocal cur, body
        if cur is None:
            return
        n, title, when, check = cur
        steps.append(Step(n, title, phase_n, phase_title, phase_goal,
                          when, "\n".join(body).strip(), check))
        cur, body = None, []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        m = RE_PHASE.match(line)
        if m:
            flush()
            phase_n, phase_title = int(m.group(1)), m.group(2)
            phase_goal, pending_goal = "", True
            continue

        m = RE_DAY.match(line)
        if m:
            flush()
            pending_goal = False
            cur = [int(m.group(1)), m.group(2), "", ""]
            seen_date = False
            continue

        if cur is not None:
            m = RE_DONE.match(line)
            if m:
                cur[3] = m.group(1)
                continue
            if not seen_date:
                m = RE_DATE.match(line)
                if m:
                    cur[2] = m.group(1)
                    seen_date = True
                    continue
            if line.strip() and not line.startswith("**Days "):
                body.append(line)
        elif pending_goal and line.strip() and not line.startswith("**Days "):
            phase_goal = (phase_goal + " " + line.strip()).strip()

    flush()
    if not steps:
        die(f"parsed no steps out of {path} - has its format changed?")
    return steps


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            warn("state file is corrupt; starting fresh (run `resync` to rebuild)")
    return {"runs": []}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE_FILE)


def completed_days(state: dict) -> set:
    return {r["day"] for r in state["runs"] if r.get("status") == "done"}


def next_step(steps: list, state: dict):
    done = completed_days(state)
    for s in steps:
        if s.n not in done:
            return s
    return None


# --------------------------------------------------------------------------
# shell helpers
# --------------------------------------------------------------------------

def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str):
    print(f"warning: {msg}", file=sys.stderr)


def info(msg: str):
    print(msg, flush=True)


def git(repo: Path, *args, check=True, capture=True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=capture, text=True)
    if check and r.returncode != 0:
        die(f"git {' '.join(args)} failed:\n{(r.stderr or r.stdout).strip()}")
    return (r.stdout or "").strip()


def notify(cfg: dict, title: str, body: str, urgent=False):
    if not flag(cfg, "NOTIFY"):
        return
    if not shutil.which("notify-send"):
        return
    subprocess.run(["notify-send", "-a", "Sixty Sessions",
                    "-u", "critical" if urgent else "normal",
                    title, body],
                   capture_output=True)


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def preflight(cfg: dict, strict: bool) -> list:
    """Return a list of problems. `strict` adds the checks a real run needs."""
    problems = []
    repo = Path(cfg["REPO_DIR"])

    if not (repo / ".git").exists():
        problems.append(f"{repo} is not a git repository")
        return problems

    try:
        agent_argv, _ = build_agent_cmd(cfg, "preflight probe")
        exe = agent_argv[0]
        if not shutil.which(exe) and not Path(exe).is_file():
            problems.append(f"AGENT_CMD runs `{exe}`, which is not on PATH")
    except ValueError as exc:
        problems.append(str(exc))

    roadmap = repo / cfg["ROADMAP_FILE"]
    if not roadmap.exists():
        problems.append(f"roadmap file missing: {roadmap}")

    if flag(cfg, "PUSH"):
        # Probe write access for real. Two traps here: `ls-remote` only proves
        # READ access, which a public repo grants to anyone; and a --dry-run
        # push at the base branch short-circuits with "Everything up-to-date"
        # without ever authenticating. Pushing --dry-run to a ref that does not
        # exist forces the auth handshake, and --dry-run guarantees no ref is
        # actually created.
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/true")
        r = subprocess.run(
            ["git", "-C", str(repo), "push", "--dry-run", "origin",
             "HEAD:refs/heads/__roadmap_preflight_probe__"],
            capture_output=True, text=True, timeout=90, env=env)
        if r.returncode != 0:
            detail = [ln for ln in (r.stderr or "").strip().splitlines() if ln.strip()]
            problems.append(
                "no push access to origin - the daily commit would be stranded "
                "on this machine. Fix with `gh auth login` (then "
                "`gh auth setup-git`), or store a credential for the remote.\n"
                f"        git said: {detail[-1] if detail else 'unknown error'}")

    if flag(cfg, "OPEN_PR"):
        if not shutil.which("gh"):
            problems.append("OPEN_PR=1 but `gh` is not installed")
        else:
            r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
            if r.returncode != 0:
                problems.append("OPEN_PR=1 but `gh auth status` fails")

    if strict:
        dirty = git(repo, "status", "--porcelain")
        if dirty:
            problems.append(
                "working tree is not clean - refusing to run so your own "
                "changes are never clobbered:\n        "
                + "\n        ".join(dirty.splitlines()[:10]))

        branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if branch != cfg["BASE_BRANCH"]:
            problems.append(
                f"on branch '{branch}', expected '{cfg['BASE_BRANCH']}'. "
                "Switch back or set BASE_BRANCH.")

    return problems


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

def build_prompt(step, cfg: dict, steps: list, state: dict) -> str:
    done = sorted(completed_days(state))
    history = ", ".join(f"day {d}" for d in done[-6:]) if done else "none yet"
    repo = cfg["REPO_DIR"]

    return textwrap.dedent(f"""\
        You are working through a 60-day roadmap for a personal quantitative
        finance project, one step per day, unattended. The full plan is in
        {cfg['ROADMAP_FILE']} at the repository root - read it for context.

        Today you implement EXACTLY ONE step: day {step.n} of {len(steps)}.

        ## Phase {step.phase_n} - {step.phase_title}
        {step.phase_goal}

        ## Day {step.n} - {step.title}

        {step.body}

        ACCEPTANCE CRITERION (the step is not finished until this is true):
        {step.check}

        ## Recently completed
        {history}

        ## How to work

        1. Read {cfg['ROADMAP_FILE']} and look at the current state of the repo
           before changing anything. Later steps depend on earlier ones; build on
           what is already there rather than rewriting it.
        2. Implement the step fully - code, tests and any documentation it needs.
           A step is done when someone could pick up the repo and see the
           acceptance criterion satisfied.
        3. Verify your own work. Run whatever is available: `make test`,
           `make lint`, `pytest`, or just execute the code. If the project has no
           test harness yet (it arrives on day 5), run the relevant code by hand
           and show that it does what the step claims.
        4. If the acceptance criterion cannot be met, stop and explain why rather
           than faking it or weakening the criterion. A failed day reported
           honestly is far more useful than a green one that lied.

        ## Hard rules

        - Do ONLY this step. Do not start the next one, do not "while I'm here"
          refactor unrelated code, do not add features the step did not ask for.
        - Never modify anything under `automation/` - that is the runner
          executing you right now.
        - Never write to `data/`, and never commit files from it. Market data
          does not belong in git.
        - Do not run `git commit`, `git push`, `git checkout`, `git reset` or
          `git rebase`. The runner owns version control and will commit your work
          for you on the correct branch.
        - Do not delete or rewrite {cfg['ROADMAP_FILE']}.
        - Work only inside {repo}.

        ## Finish with

        A short report, in this exact shape:

        SUMMARY: <one line, imperative mood, under 70 characters - this becomes
        the commit subject>
        STATUS: <COMPLETE or BLOCKED>
        VERIFIED: <the exact command(s) you ran and their outcome>
        NOTES: <anything the next session needs to know - at most 3 lines>
        """)


# --------------------------------------------------------------------------
# running the agent
# --------------------------------------------------------------------------

def build_agent_cmd(cfg: dict, prompt: str):
    """Turn AGENT_CMD into an argv list. Returns (argv, stdin_text).

    The template is split into words *before* the placeholders are filled in,
    so a prompt full of quotes, newlines and backslashes always arrives as
    exactly one argument and never has to survive a shell. Placeholders:
    {prompt} {model} {tools} {denied} {repo}. An agent that wants its prompt
    on stdin instead just leaves {prompt} out of the template.

    Raises ValueError if the template is unusable, so preflight can report it
    as a problem rather than dying mid-run.
    """
    template = str(cfg.get("AGENT_CMD", "")).strip()
    if not template:
        raise ValueError("AGENT_CMD is empty - there is no agent to run")
    try:
        words = shlex.split(template)
    except ValueError as exc:
        raise ValueError(f"AGENT_CMD does not parse as a command: {exc}") from exc
    if not words:
        raise ValueError("AGENT_CMD is empty - there is no agent to run")

    values = {
        "{prompt}": prompt,
        "{model}": cfg.get("MODEL", ""),
        "{tools}": cfg.get("ALLOWED_TOOLS", ""),
        "{denied}": cfg.get("DENIED_TOOLS", ""),
        "{repo}": cfg.get("REPO_DIR", ""),
    }
    argv = []
    for word in words:
        for key, val in values.items():
            word = word.replace(key, val)
        argv.append(word)

    return argv, (None if "{prompt}" in template else prompt)


def run_agent(cfg: dict, prompt: str, log_path: Path):
    """Return (ok, result_text, failure_reason)."""
    try:
        cmd, stdin_text = build_agent_cmd(cfg, prompt)
    except ValueError as exc:
        return False, "", str(exc)

    agent = Path(cmd[0]).name
    # The prompt is thousands of words; keep the log readable.
    shown = " ".join("<prompt>" if w == prompt else shlex.quote(w) for w in cmd)

    timeout = int(cfg["STEP_TIMEOUT"])
    started = time.time()
    try:
        r = subprocess.run(cmd, cwd=cfg["REPO_DIR"], capture_output=True,
                           text=True, timeout=timeout, input=stdin_text)
    except subprocess.TimeoutExpired:
        log_path.write_text(f"# agent: {shown}\nTIMED OUT after {timeout}s\n",
                            encoding="utf-8")
        return False, "", f"agent exceeded STEP_TIMEOUT ({timeout}s)"
    except (FileNotFoundError, PermissionError) as exc:
        return False, "", f"could not run agent `{cmd[0]}`: {exc}"

    elapsed = int(time.time() - started)
    log_path.write_text(
        f"# agent: {shown}\n# exit={r.returncode} elapsed={elapsed}s\n\n"
        f"## stdout\n{r.stdout}\n\n## stderr\n{r.stderr}\n", encoding="utf-8")

    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return False, "", f"{agent} exited {r.returncode}: {tail[-1] if tail else 'no output'}"

    # Agents that print plain text fall through the JSONDecodeError below, so
    # a non-JSON CLI needs no code change here - only a different AGENT_CMD.
    text = r.stdout
    try:
        payload = json.loads(r.stdout)
        text = payload.get("result") or payload.get("text") or r.stdout
        if payload.get("is_error"):
            return False, text, f"{agent} reported an error result"
    except json.JSONDecodeError:
        pass  # fall back to the raw text

    return True, text, ""


def parse_report(text: str) -> dict:
    out = {"summary": "", "status": "", "verified": "", "notes": ""}
    for key in out:
        m = re.search(rf"^{key.upper()}:\s*(.+?)\s*$", text, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    return out


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

PROTECTED = ("automation/", "data/")


def verify(cfg: dict, repo: Path):
    """Check the agent stayed in bounds and the project still builds."""
    changed = [ln[3:].strip() for ln in
               git(repo, "status", "--porcelain").splitlines()]
    changed = [c.split(" -> ")[-1].strip('"') for c in changed]

    if not changed:
        return False, "the agent changed no files", []

    trespass = [c for c in changed if any(c.startswith(p) for p in PROTECTED)]
    if trespass:
        return False, "agent modified protected paths: " + ", ".join(trespass), changed

    # Only gate on a test suite once one exists (day 5 creates it).
    makefile = repo / "Makefile"
    if makefile.exists():
        mk = makefile.read_text(encoding="utf-8", errors="replace")
        for target in ("test", "lint"):
            if re.search(rf"^{target}:", mk, re.MULTILINE):
                r = subprocess.run(["make", target], cwd=repo,
                                   capture_output=True, text=True, timeout=1800)
                if r.returncode != 0:
                    tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
                    return False, f"`make {target}` failed:\n" + "\n".join(tail), changed

    return True, "", changed


# --------------------------------------------------------------------------
# landing the day on the base branch
# --------------------------------------------------------------------------

def try_git(repo: Path, *args):
    """Run git without dying. Returns (ok, last line of output)."""
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    detail = (r.stderr or r.stdout or "").strip().splitlines()
    return r.returncode == 0, (detail[-1] if detail else "")


def merge_into_base(cfg: dict, repo: Path, step) -> tuple:
    """Land a finished day on the base branch locally, so no PR is needed.

    Fast-forward if the base has not moved, a merge commit if it has. On any
    failure the base is left exactly as it was and the work stays on its own
    branch - the same place a failed day leaves it - so nothing is ever lost.
    Returns (merged, reason).
    """
    base = cfg["BASE_BRANCH"]

    ok, err = try_git(repo, "checkout", base)
    if not ok:
        return False, f"could not check out {base}: {err}"

    ok, err = try_git(repo, "merge", "--ff-only", step.branch)
    if not ok:
        # Base moved underneath us: record the join explicitly instead.
        ok, err = try_git(repo, "merge", "--no-ff", "--no-edit", step.branch)
        if not ok:
            try_git(repo, "merge", "--abort")
            try_git(repo, "checkout", step.branch)
            return False, err

    try_git(repo, "branch", "-d", step.branch)
    return True, ""


# --------------------------------------------------------------------------
# tracker sync (best effort)
# --------------------------------------------------------------------------

def tick_artifact(cfg: dict, step, note: str):
    url = str(cfg.get("ARTIFACT_URL", "")).strip()
    if not url:
        return
    # This one is genuinely Claude-specific - it drives the Artifact tool - and
    # is unrelated to AGENT_CMD. If you swap the agent out, drop ARTIFACT_URL.
    if not shutil.which("claude"):
        return
    payload = json.dumps({"done": {str(step.n): date.today().isoformat()},
                          "notes": {str(step.n): note[:200]}})
    prompt = (
        "Use the Artifact tool once, with no other tools and no commentary: "
        f'action "write_db", url "{url}", db_op "update", collection "progress", '
        f'doc_id "state", data {payload}. Then reply with just OK.')
    try:
        subprocess.run(["claude", "-p", prompt, "--output-format", "json",
                        "--model", "haiku", "--permission-mode", "acceptEdits",
                        "--permission-prompts", "none",
                        "--allowedTools", "Artifact"],
                       cwd=cfg["REPO_DIR"], capture_output=True,
                       text=True, timeout=240)
    except Exception as exc:  # never let the tracker break the run
        warn(f"could not tick the tracker artifact: {exc}")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_status(cfg, args):
    repo = Path(cfg["REPO_DIR"])
    steps = parse_roadmap(repo / cfg["ROADMAP_FILE"])
    state = load_state()
    done = completed_days(state)
    nxt = next_step(steps, state)

    pct = round(100 * len(done) / len(steps))
    info(f"Sixty Sessions - {len(done)}/{len(steps)} complete ({pct}%)")
    info("")
    for phase in sorted({s.phase_n for s in steps}):
        ps = [s for s in steps if s.phase_n == phase]
        d = sum(1 for s in ps if s.n in done)
        bar = "".join("#" if s.n in done else "." for s in ps)
        info(f"  Phase {phase}  [{bar}]  {d}/{len(ps)}  {ps[0].phase_title}")
    info("")
    if nxt:
        info(f"Next:  Day {nxt.n:02d} - {nxt.title}")
        info(f"       {nxt.check}")
    else:
        info("All sixty sessions complete.")

    failures = [r for r in state["runs"] if r.get("status") != "done"]
    if failures:
        info("")
        info(f"{len(failures)} failed attempt(s):")
        for r in failures[-5:]:
            info(f"  day {r['day']:02d}  {r.get('at', '')}  {str(r.get('reason', ''))[:90]}")
    return 0


def cmd_preflight(cfg, args):
    problems = preflight(cfg, strict=args.strict)
    if not problems:
        info("preflight: all clear")
        return 0
    info("preflight found problems:")
    for p in problems:
        info(f"  - {p}")
    return 1


def cmd_resync(cfg, args):
    """Rebuild state from commit trailers, for when the state file is lost."""
    repo = Path(cfg["REPO_DIR"])
    log = git(repo, "log", "--all", "--format=%H%x1f%cI%x1f%B%x1e")
    state = {"runs": []}
    seen = set()
    for entry in log.split("\x1e"):
        if not entry.strip():
            continue
        parts = entry.strip().split("\x1f")
        sha, iso, body = (parts + ["", ""])[:3]
        m = re.search(rf"^{TRAILER}\s*(\d+)", body, re.MULTILINE)
        if not m:
            continue
        day = int(m.group(1))
        if day in seen:
            continue
        seen.add(day)
        lines = body.strip().splitlines()
        state["runs"].append({"day": day, "status": "done", "at": iso,
                              "commit": sha[:10],
                              "summary": lines[0] if lines else ""})
    state["runs"].sort(key=lambda r: r["day"])
    save_state(state)
    info(f"rebuilt state from git: {len(state['runs'])} completed day(s)")
    return 0


def cmd_run(cfg, args):
    repo = Path(cfg["REPO_DIR"])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # One run at a time, ever.
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        die("another run is already in progress")

    steps = parse_roadmap(repo / cfg["ROADMAP_FILE"])
    state = load_state()

    if args.day:
        step = next((s for s in steps if s.n == args.day), None)
        if not step:
            die(f"no day {args.day} in the roadmap")
        if step.n in completed_days(state) and not args.force:
            die(f"day {args.day} is already done (use --force to redo it)")
    else:
        step = next_step(steps, state)
        if not step:
            info("all sixty sessions complete - nothing to do")
            return 0

    today = date.today().isoformat()
    if not args.force and not args.day:
        if any(str(r.get("at", "")).startswith(today) for r in state["runs"]):
            info(f"a step already ran today ({today}) - skipping")
            return 0

    prompt = build_prompt(step, cfg, steps, state)

    if args.dry_run:
        info(f"--- would run day {step.n}: {step.title} ---")
        info(f"branch: {step.branch}")
        info("")
        info(prompt)
        return 0

    problems = preflight(cfg, strict=True)
    if problems:
        msg = "; ".join(p.splitlines()[0] for p in problems)
        for p in problems:
            warn(p)
        # Record the attempt. Without this the systemd retry half an hour later
        # finds nothing in state, runs the same blocked preflight and notifies
        # again - every thirty minutes, for as long as the problem goes
        # unfixed. With it the retry hits the "already ran today" guard and
        # exits quietly, so you are told once a day instead.
        state["runs"].append({
            "day": step.n, "status": "blocked",
            "at": datetime.now().isoformat(timespec="seconds"),
            "title": step.title, "reason": msg, "log": None,
        })
        save_state(state)
        notify(cfg, "Roadmap runner blocked", msg, urgent=True)
        return 1

    info(f"Day {step.n:02d} - {step.title}")

    # Start from a current base branch.
    git(repo, "fetch", "origin", "--quiet", check=False)
    git(repo, "pull", "--ff-only", "--quiet", check=False)

    branch_mode = cfg["BRANCH_MODE"]
    if branch_mode == "branch":
        git(repo, "checkout", "-B", step.branch)
        info(f"  branch: {step.branch}")

    try:
        log_path = LOG_DIR / f"day-{step.n:02d}-{datetime.now():%Y%m%d-%H%M%S}.log"
        info(f"  log:    {log_path}")
        info("  running agent...")

        ok, text, reason = run_agent(cfg, prompt, log_path)
        report = parse_report(text)
        changed = []

        if ok and report.get("status", "").upper() == "BLOCKED":
            ok = False
            reason = "agent reported BLOCKED: " + (report.get("notes") or "no reason given")

        if ok:
            ok, vreason, changed = verify(cfg, repo)
            if not ok:
                reason = vreason

        if not ok:
            record_failure(cfg, state, step, reason, log_path, branch_mode, repo)
            return 1

        summary = report.get("summary") or f"implement day {step.n}"
        subject = f"day {step.n:02d}: {summary}"[:72]
        message = "\n".join([
            subject, "",
            f"Roadmap day {step.n} of {len(steps)} - {step.title}",
            f"Phase {step.phase_n}: {step.phase_title}",
            "",
            f"Acceptance: {step.check}",
            f"Verified:   {report.get('verified') or 'see runner log'}",
        ] + ([f"Notes:      {report['notes']}"] if report.get("notes") else []) + [
            "",
            f"{TRAILER} {step.n}",
            f"Runner-Log: {log_path.name}",
            "",
            "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
        ])

        git(repo, "add", "-A", "--", ".", ":!data")
        git(repo, "commit", "-m", message)
        sha = git(repo, "rev-parse", "--short", "HEAD")
        info(f"  committed {sha}: {subject}")

        merged = False
        if branch_mode == "branch" and flag(cfg, "AUTO_MERGE"):
            merged, mreason = merge_into_base(cfg, repo, step)
            if merged:
                info(f"  merged into {cfg['BASE_BRANCH']}")
            else:
                warn(f"auto-merge failed, work stays on {step.branch}: {mreason}")

        pushed = False
        if flag(cfg, "PUSH"):
            target = step.branch if branch_mode == "branch" and not merged else cfg["BASE_BRANCH"]
            r = subprocess.run(["git", "-C", str(repo), "push", "-u", "origin", target],
                               capture_output=True, text=True)
            pushed = r.returncode == 0
            if pushed:
                info(f"  pushed to origin/{target}")
            else:
                detail = (r.stderr or "").strip().splitlines()
                warn("push failed: " + (detail[-1] if detail else "unknown error"))

        pr_url = ""
        if pushed and not merged and branch_mode == "branch" and flag(cfg, "OPEN_PR"):
            r = subprocess.run(
                ["gh", "pr", "create", "--fill", "--base", cfg["BASE_BRANCH"],
                 "--head", step.branch, "--title", subject],
                cwd=repo, capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                pr_url = r.stdout.strip().splitlines()[-1]
                info(f"  PR: {pr_url}")

        if branch_mode == "branch":
            git(repo, "checkout", cfg["BASE_BRANCH"], check=False)

        state["runs"].append({
            "day": step.n, "status": "done",
            "at": datetime.now().isoformat(timespec="seconds"),
            "title": step.title, "summary": summary, "commit": sha,
            "branch": step.branch if branch_mode == "branch" and not merged
                      else cfg["BASE_BRANCH"],
            "merged": merged,
            "pushed": pushed, "pr": pr_url, "log": log_path.name,
            "changed": len(changed),
        })
        save_state(state)

        tick_artifact(cfg, step, summary)
        remaining = len(steps) - len(completed_days(state))

        # "Done" must not imply the work is safely off this machine.
        stranded = flag(cfg, "PUSH") and not pushed
        if stranded:
            warn("committed but NOT pushed - the work is only on this machine")
            notify(cfg, f"Day {step.n:02d} done, NOT PUSHED - {step.title}",
                   f"{summary}\nThe commit is only on this machine.\n"
                   f"{remaining} sessions left", urgent=True)
        else:
            notify(cfg, f"Day {step.n:02d} done - {step.title}",
                   f"{summary}\n{remaining} sessions left")
        info(f"  done. {remaining} sessions remaining.")
        return 0
    except BaseException as exc:
        # Any unplanned exit at all: a git command that called die(), a bug in
        # here, Ctrl-C, the machine going down mid-run. Without this the repo is
        # left sitting on the day branch with uncommitted changes and nothing
        # written to state, and strict preflight then blocks every remaining
        # evening with nobody watching.
        reason = f"runner exited unexpectedly: {type(exc).__name__}: {exc}".strip()
        try:
            rescue(cfg, state, step, reason, log_path, branch_mode, repo)
        except BaseException as inner:   # never mask the original fault
            warn(f"cleanup after that failure also failed: {inner}")
        raise


def day_in_git(repo: Path, day: int) -> bool:
    """Has this day's work already been committed somewhere in the repo?"""
    out = git(repo, "log", "--all", "-n", "300", "--format=%B%x1e", check=False)
    return bool(re.search(rf"^{TRAILER}\s*{day}\b", out, re.MULTILINE))


def rescue(cfg, state, step, reason, log_path, branch_mode, repo):
    """Put the repo back the way preflight expects after an unplanned exit."""
    warn(f"day {step.n} ended unexpectedly: {reason}")

    # Whatever else happens here, the tree has to end up clean and on the base
    # branch, or strict preflight blocks every remaining evening. A leftover
    # dirty tree therefore always goes down the record_failure path, which
    # parks it as a WIP commit - even if the day's own commit did land.
    dirty = bool(git(repo, "status", "--porcelain", check=False))

    if day_in_git(repo, step.n) and not dirty:
        # The commit landed; only the bookkeeping was lost. `resync` would
        # reach the same conclusion, so record it now rather than let tomorrow
        # redo a day that is already finished.
        sha = git(repo, "rev-parse", "--short", "HEAD", check=False)
        if branch_mode == "branch":
            git(repo, "checkout", cfg["BASE_BRANCH"], check=False)
        state["runs"].append({
            "day": step.n, "status": "done",
            "at": datetime.now().isoformat(timespec="seconds"),
            "title": step.title,
            "summary": "recovered after an unexpected exit",
            "commit": sha,
            "branch": step.branch if branch_mode == "branch" else cfg["BASE_BRANCH"],
            "merged": False, "pushed": False, "pr": "",
            "log": log_path.name, "changed": 0,
        })
        save_state(state)
        notify(cfg, f"Day {step.n:02d} committed, but the runner crashed",
               "The work is in git. Check it and push it by hand.", urgent=True)
        return

    record_failure(cfg, state, step, reason, log_path, branch_mode, repo)


def record_failure(cfg, state, step, reason, log_path, branch_mode, repo):
    warn(f"day {step.n} failed: {reason}")
    # Keep the partial work for inspection, but never on the base branch.
    if branch_mode == "branch":
        if git(repo, "status", "--porcelain"):
            git(repo, "add", "-A", "--", ".", ":!data", check=False)
            git(repo, "commit", "-m",
                f"WIP day {step.n:02d}: {step.title} (FAILED)\n\n{reason}\n\n"
                f"Attempt-For-Day: {step.n}", check=False)
            info(f"  incomplete work left on branch {step.branch} for review")
        git(repo, "checkout", cfg["BASE_BRANCH"], check=False)
    else:
        git(repo, "stash", "push", "-u", "-m",
            f"failed day {step.n}", check=False)
        info("  partial work stashed (BRANCH_MODE=main)")

    state["runs"].append({
        "day": step.n, "status": "failed",
        "at": datetime.now().isoformat(timespec="seconds"),
        "title": step.title, "reason": reason, "log": log_path.name,
        "branch": step.branch if branch_mode == "branch" else None,
    })
    save_state(state)
    notify(cfg, f"Day {step.n:02d} FAILED - {step.title}",
           reason.splitlines()[0][:180], urgent=True)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run one roadmap step a day.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show progress and the next step")

    pf = sub.add_parser("preflight", help="check the machine is set up")
    pf.add_argument("--strict", action="store_true",
                    help="also require a clean tree on the base branch")

    sub.add_parser("resync", help="rebuild state from git history")

    r = sub.add_parser("run", help="implement the next step")
    r.add_argument("--day", type=int, help="run a specific day instead of the next")
    r.add_argument("--dry-run", action="store_true", help="print the prompt and stop")
    r.add_argument("--force", action="store_true", help="run even if one already ran today")

    args = p.parse_args()
    cfg = load_config()

    if args.cmd == "status":
        return cmd_status(cfg, args)
    if args.cmd == "preflight":
        return cmd_preflight(cfg, args)
    if args.cmd == "resync":
        return cmd_resync(cfg, args)
    if args.cmd == "run":
        return cmd_run(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
