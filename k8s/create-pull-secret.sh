#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="mlops-torch-triton-dgx-pipeline"
SECRET_NAME="dockerhub-pull-secret"
USERNAME="${DOCKERHUB_USERNAME:-aaroncody}"

if [ -z "${DOCKERHUB_TOKEN:-}" ]; then
  read -rsp "Docker Hub token: " DOCKERHUB_TOKEN
  echo
fi

kubectl --context minikube create secret docker-registry "$SECRET_NAME" \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username="$USERNAME" \
  --docker-password="$DOCKERHUB_TOKEN" \
  --namespace="$NAMESPACE" \
  --dry-run=client -o yaml \
  | kubectl --context minikube apply -f -

echo "Secret '$SECRET_NAME' ready in namespace '$NAMESPACE'"
