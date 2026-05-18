#!/bin/bash
set -euo pipefail

TOKEN="${1:?Usage: launch.sh <RUNNER_TOKEN>}"

# Detect arch
case "$(uname -m)" in
  x86_64)  RUNNER_NAME=MSIWSL2;  RUNNER_LABELS=msi-wsl2  ;;
  aarch64) RUNNER_NAME=DGXSPARK; RUNNER_LABELS=dgx-spark ;;
  *) echo "Unknown arch: $(uname -m)"; exit 1 ;;
esac

IMAGE=ghcr.io/miramar-labs/github-runner:latest
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

echo "Runner: $RUNNER_NAME ($RUNNER_LABELS) on $(uname -m)"
echo "Image:  $IMAGE"

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q '^github-runner$'; then
  echo "Stopping existing github-runner container..."
  docker rm -f github-runner
fi

docker pull "$IMAGE"

docker run -d --restart unless-stopped \
  --name github-runner \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --group-add "$DOCKER_GID" \
  -e REPO_URL=https://github.com/miramar-labs/github-actions-hello \
  -e RUNNER_TOKEN="$TOKEN" \
  -e RUNNER_NAME="$RUNNER_NAME" \
  -e RUNNER_LABELS="$RUNNER_LABELS" \
  "$IMAGE"

echo "Started. Logs: docker logs -f github-runner"
