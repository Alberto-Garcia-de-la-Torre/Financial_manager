#!/usr/bin/env bash
# Install the Sixty Sessions daily timer as a systemd user unit.
#
#   ./install.sh              install and enable
#   ./install.sh --uninstall  remove it again
#
# Nothing here needs root except `loginctl enable-linger`, which is what
# lets the timer fire when you are not logged in. It will ask if needed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE=finmgr-roadmap.service
TIMER=finmgr-roadmap.timer

say()  { printf '  %s\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- uninstall
if [[ "${1:-}" == "--uninstall" ]]; then
  step "Removing the timer"
  systemctl --user disable --now "$TIMER" 2>/dev/null || true
  rm -f "$UNIT_DIR/$SERVICE" "$UNIT_DIR/$TIMER"
  systemctl --user daemon-reload
  say "removed. State and logs are kept in ~/.local/state/finmgr-roadmap/"
  exit 0
fi

# ------------------------------------------------------------------- config
[[ -f "$HERE/config.env" ]] || fail "config.env not found next to install.sh"

get() { grep -E "^$1=" "$HERE/config.env" | tail -1 | cut -d= -f2- | tr -d '"'"'"' '; }
REPO_DIR="$(get REPO_DIR)"
RUN_TIME="$(get RUN_TIME)"
: "${REPO_DIR:?REPO_DIR missing from config.env}"
: "${RUN_TIME:?RUN_TIME missing from config.env}"

[[ "$RUN_TIME" =~ ^[0-9]{2}:[0-9]{2}$ ]] || fail "RUN_TIME must look like 19:30, got '$RUN_TIME'"
[[ -d "$REPO_DIR/.git" ]] || fail "REPO_DIR '$REPO_DIR' is not a git repository"

step "Checking the machine"
command -v claude  >/dev/null || fail "claude is not on PATH"
command -v python3 >/dev/null || fail "python3 is not on PATH"
say "claude  $(claude --version 2>/dev/null | head -1)"
say "python  $(python3 --version)"
say "repo    $REPO_DIR"
say "time    $RUN_TIME daily ($(timedatectl show -p Timezone --value 2>/dev/null || echo local))"

# ------------------------------------------------------------------ linger
step "Checking unattended operation"
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo no)" == "yes" ]]; then
  say "lingering is on - the timer runs even when you are logged out"
else
  say "lingering is OFF: user timers only fire while you are logged in."
  say "Turning it on (this needs your password):"
  if sudo loginctl enable-linger "$USER"; then
    say "lingering enabled"
  else
    say "could not enable lingering - the timer will still work whenever"
    say "you are logged in, and Persistent=true will catch up missed runs."
  fi
fi

# ------------------------------------------------------------------- units
step "Installing units into $UNIT_DIR"
mkdir -p "$UNIT_DIR"
sed "s|__REPO_DIR__|$REPO_DIR|g" "$HERE/systemd/$SERVICE" > "$UNIT_DIR/$SERVICE"
sed "s|__RUN_TIME__|$RUN_TIME|g"  "$HERE/systemd/$TIMER"   > "$UNIT_DIR/$TIMER"
chmod +x "$HERE/roadmap_runner.py"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER"
say "installed and enabled"

# ---------------------------------------------------------------- preflight
step "Runner preflight"
if python3 "$HERE/roadmap_runner.py" preflight; then
  :
else
  say ""
  say "Fix the above before the first run, or the timer will just report"
  say "the same problem to your desktop every evening."
fi

step "Done"
systemctl --user list-timers "$TIMER" --no-pager 2>/dev/null | sed 's/^/  /'
cat <<EOF

  Useful from here:
    automation/roadmap_runner.py status          progress and next step
    automation/roadmap_runner.py run --dry-run   see tomorrow's prompt
    systemctl --user start finmgr-roadmap        run one step right now
    journalctl --user -u finmgr-roadmap -f       watch it work
    ./install.sh --uninstall                     stop all of this

EOF
