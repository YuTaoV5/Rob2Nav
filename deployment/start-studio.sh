#!/bin/bash
set -euo pipefail
CONF=${ROB2NAV_SUPERVISOR_CONFIG:-/root/gpufree-data/deployment/rob2nav-supervisord.conf}
if ! supervisorctl -c "$CONF" pid >/dev/null 2>&1; then
  supervisord -c "$CONF"
fi
for name in rob2nav-studio rob2nav-paw rob2nav-navigation rob2nav-scan-bridge rob2nav-scan; do
  if ! supervisorctl -c "$CONF" status "$name" | grep -q RUNNING; then
    supervisorctl -c "$CONF" start "$name"
  fi
done
supervisorctl -c "$CONF" status
