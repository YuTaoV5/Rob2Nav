#!/usr/bin/env bash
set -e
# Ignore automatic Foxy/Conda/Python overlays inherited from a factory terminal.
exec env -i HOME="$HOME" USER="${USER:-}" TERM="${TERM:-xterm}" \
  DISPLAY="${DISPLAY:-}" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash --noprofile --rcfile "$HOME/Rob2Nav/deployment/go2edu/env.noetic" -i
