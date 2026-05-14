#!/usr/bin/env bash
set -e

# Ensure the matlab container user (UID 1001) can write to files owned by the current user,
# skipping .git and any files owned by other users (e.g. root-owned docker artifacts)
find "$(dirname "$0")" -not -path '*/.git/*' -not -path '*/.git' \
    -user "$(id -u)" -exec chmod o+w {} +

case "${1:-browser}" in
  cli)
    docker compose -f docker-compose.matlab.yml --profile cli run --rm matlab-cli
    ;;
  browser)
    docker compose -f docker-compose.matlab.yml --profile browser up matlab-browser
    ;;
  *)
    echo "Usage: $0 [cli|browser]"
    exit 1
    ;;
esac
