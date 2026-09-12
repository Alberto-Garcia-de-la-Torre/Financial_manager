# Sixty Sessions — the daily runner

Every evening at a fixed time this picks the next unfinished step out of
`ROADMAP.md`, hands it to Claude Code in headless mode, checks the result,
and commits it. Sixty days, sixty commits, no supervision required.

```
automation/
├── config.env            everything you'd want to change
├── roadmap_runner.py     the runner itself
├── install.sh            installs/removes the systemd timer
└── systemd/              unit templates
```

## Install

```bash
cd automation
./install.sh
```

That validates the machine, installs a systemd **user** timer, enables
lingering (so it fires when you're logged out), and runs a preflight.

## Daily commands

```bash
automation/roadmap_runner.py status          # progress and what's next
automation/roadmap_runner.py run --dry-run   # see tonight's prompt, change nothing
automation/roadmap_runner.py run             # run the next step now
automation/roadmap_runner.py run --day 12    # run a specific step
automation/roadmap_runner.py preflight       # check the setup
automation/roadmap_runner.py resync          # rebuild state from git history

systemctl --user start finmgr-roadmap        # trigger tonight's run early
systemctl --user list-timers finmgr-roadmap  # when does it next fire?
journalctl --user -u finmgr-roadmap -f       # watch it work
./install.sh --uninstall                     # stop all of it
```

## What one run actually does

1. **Takes a lock.** Two runs can never overlap.
2. **Refuses to start on a dirty tree.** If you have uncommitted work, the
   run aborts rather than sweeping your changes into its commit.
3. **Fast-forwards** the base branch from `origin`.
4. **Cuts a branch**, `roadmap/day-NN-slug`.
5. **Runs the agent** on exactly one step, with a wall-clock timeout.
6. **Verifies** the result (see below).
7. **Commits** with a structured message carrying a `Roadmap-Day: N` trailer.
8. **Pushes** the branch, optionally opening a PR.
9. **Records** the run, notifies your desktop, and ticks the day off in the
   tracker artifact.

## The safety rails

These exist because the thing runs unattended and writes code.

| Rail | Behaviour |
|---|---|
| Dirty working tree | Run aborts before touching anything |
| Wrong branch | Run aborts |
| No push access | Caught at preflight, not at 19:30 |
| Agent edits `automation/` | Run fails — it can't rewrite its own runner |
| Agent writes to `data/` | Run fails — market data stays out of git |
| Agent changes nothing | Run fails rather than making an empty commit |
| `make test` / `make lint` fails | Run fails, nothing is pushed |
| Agent reports `BLOCKED` | Recorded as failed, work left on the branch |
| Step overruns `STEP_TIMEOUT` | Abandoned, marked failed |
| Two runs at once | Second one exits immediately |
| Already ran today | Second invocation is a no-op |

Git verbs are denied to the agent outright (`--disallowedTools` covers
`git commit`, `push`, `checkout`, `reset`, `rebase`) — the runner owns
version control. Anything not on the allowlist is **denied automatically**
rather than prompting, so an unattended run can never hang waiting for
input it will never get.

A failed day is never silently skipped: partial work is committed to its
branch as `WIP ... (FAILED)` for you to inspect, the base branch is left
untouched, and `status` lists it. Fix it by hand, or re-run with
`run --day N --force`.

## Configuration

All in `config.env`:

| Key | Default | Notes |
|---|---|---|
| `RUN_TIME` | `19:30` | Local time. Re-run `install.sh` after changing. |
| `AGENT_CMD` | Claude Code | The whole agent, as one command line. See below. |
| `MODEL` | `opus` | Substituted into `AGENT_CMD` as `{model}`. |
| `STEP_TIMEOUT` | `3600` | Seconds for one step. |
| `BRANCH_MODE` | `branch` | `branch` = one branch per day (reviewable). `main` = commit straight to main. |
| `PUSH` | `1` | Push after committing. |
| `OPEN_PR` | `0` | Needs a working `gh auth status`. |
| `ALLOWED_TOOLS` | see file | Everything else is denied automatically. `{tools}`. |
| `DENIED_TOOLS` | git commands | Keeps the agent off version control. `{denied}`. |
| `ARTIFACT_URL` | set | Tracker to tick off. Blank to skip. Claude-only. |

## Using a different agent

Nothing outside `AGENT_CMD` knows which agent runs. The roadmap parsing,
`verify()`, the git and merge logic, the state file and the timer are all
agent-agnostic, and the contract with the agent is plain text in, plain text
out — a prompt on one side, a `SUMMARY:` / `STATUS:` / `VERIFIED:` / `NOTES:`
report on the other.

So switching is a one-line edit to `config.env`:

```bash
AGENT_CMD=gemini --yolo -p {prompt}
```

Placeholders (`{prompt} {model} {tools} {denied} {repo}`) are filled in
*after* the line is split into words, so the prompt is always exactly one
argument however many quotes and newlines it contains. Omit `{prompt}` and
it goes to the agent's stdin instead. Output is parsed as JSON if it is
JSON, otherwise taken as plain text.

Two things to know before you rely on it:

- It must be an **agentic** CLI — one that edits files and runs commands by
  itself, unattended. A chat model cannot do this job.
- `DENIED_TOOLS` is Claude Code syntax. Another agent will want its own, or
  have no equivalent. What still holds without it: the prompt's "Hard rules",
  and `verify()`, which refuses any day that touched `automation/` or `data/`
  or left `make test` red.

## State

Progress lives at `~/.local/state/finmgr-roadmap/state.json`, **outside the
repo** — so it never conflicts with a branch or a merge. Agent transcripts
go to `~/.local/state/finmgr-roadmap/logs/day-NN-*.log`.

Losing that file costs you nothing: `resync` rebuilds it by reading the
`Roadmap-Day:` trailers out of the git history.

## Things that will bite you

**Lingering.** systemd user timers only run while you're logged in unless
lingering is on. `install.sh` enables it; check with
`loginctl show-user $USER -p Linger`.

**Missed runs.** `Persistent=true`, so a laptop that was closed at 19:30
runs the step when it next wakes. It won't try to catch up on several days
at once — one step per day, by design.

**Push credentials.** `gh auth status` and `git push` use different
credentials. Preflight probes the one that matters by doing a `--dry-run`
push to a ref that doesn't exist, which forces a real auth handshake
without creating anything. A plain `ls-remote` would pass on a public
repo while the evening push still failed.

**Review the output.** This writes real code unsupervised. `BRANCH_MODE=branch`
is the default so nothing lands on `main` without you looking at it. If you
switch to `main`, you've given up that gate.
