# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

An MLOps pipeline that fine-tunes DistilBERT for text classification (IMDB sentiment), tracks experiments with MLflow, and serves the model on minikube via NVIDIA Triton Inference Server. Training and deployment both run on a DGX Station self-hosted GitHub Actions runner with GPU access.

## Repository Structure

```
.github/workflows/
  ml-train-test.yaml    # Build training image and run pytest (entry point)
  ml-train.yaml         # GPU training on DGX — exports model.onnx as GitHub artifact
  ml-deploy.yaml        # Build Triton serving image, push to Docker Hub, deploy to minikube
ml/
  train.py              # DistilBERT fine-tune on IMDB, MLflow logging, ONNX export
  test_train.py         # Unit tests for tokenize_batch, evaluate, ONNX export
  requirements.txt      # Local dev dependencies (CPU torch + pytest)
  Dockerfile.train      # GPU training image (nvcr.io/nvidia/pytorch:25.03-py3)
  Dockerfile.serve      # Triton serving image — model.onnx baked in at build time
  triton_config.pbtxt   # Triton model config: ONNX Runtime backend, input_ids + attention_mask → logits
  output/               # Generated at runtime — model.onnx (gitignored)
k8s/
  triton.yaml           # Namespace mlops-torch-triton-dgx-pipeline, Deployment triton, ClusterIP Service
```

## Workflow

Three workflows chain via `workflow_run` (each triggers the next on success):

```
ML Train Test — push to ml/ or workflow_dispatch (runner: dgx, ARM64)
  ├── pytest ml/test_train.py
  └── docker build ml-trainer image

ML Train — triggered by ML Train Test success; or workflow_dispatch (runner: dgx, ARM64, GPU)
  ├── docker build ml-trainer image
  ├── docker run --gpus all --network host → DistilBERT fine-tune + MLflow logging
  ├── export model.onnx via named Docker volume → runner filesystem
  └── upload onnx-model artifact (30-day retention)

ML Deploy — triggered by ML Train success (runner: dgx, ARM64)
  ├── download onnx-model artifact
  ├── docker build Triton serving image (model.onnx baked in)
  ├── push to Docker Hub (:latest + commit SHA tag)
  └── kubectl apply → minikube, rollout wait 300s
```

**ML Train manual dispatch inputs:**
- `epochs` (default: `3`) — training epochs
- `experiment` (default: `text-classifier`) — MLflow experiment name

**ONNX handoff:** a named Docker volume (`onnx-$RUN_ID`) passes `model.onnx` from the GPU container back to the runner, then `alpine cat` extracts it. The volume is deleted after extraction.

## Model

| Property | Value |
|---|---|
| Base model | `distilbert-base-uncased` (HuggingFace) |
| Task | Binary sentiment classification |
| Dataset | `imdb` from HuggingFace `datasets` — auto-downloaded |
| Sequence length | 128 tokens (pad/truncate) |
| ONNX inputs | `input_ids` (INT64, [batch, 128]), `attention_mask` (INT64, [batch, 128]) |
| ONNX output | `logits` (FP32, [batch, 2]) |

## MLflow

MLflow is a persistent service on the DGX host (managed separately). Training containers use `--network host` so they can reach it at `http://localhost:5000`. `MLFLOW_TRACKING_URI` is set via a GitHub org variable.

To view experiment tracking, open the MLflow UI via SSH tunnel:

```bash
ssh -L 5000:localhost:5000 aaron@spark-79b7.local
# then browse http://localhost:5000
```

## minikube

The DGX host runs a minikube cluster with the dashboard enabled. All `kubectl` commands in the deploy workflow use `--context minikube` explicitly.

| Resource | Value |
|---|---|
| Cluster | minikube on `spark-79b7.local` |
| Namespace | `mlops-torch-triton-dgx-pipeline` |
| Dashboard | `minikube dashboard` or `minikube dashboard --url` |
| Image registry | Docker Hub |

## GitHub Secrets and Variables

| Name | Scope | Type | Description |
|---|---|---|---|
| `DOCKERHUB_USERNAME` | org | Secret | Docker Hub username for pushing the Triton serving image |
| `DOCKERHUB_TOKEN` | org | Secret | Docker Hub access token |

## Runners

| Runner | Label | Host | Used for |
|---|---|---|---|
| NVIDIA DGX Spark 128GB | `dgx` | `spark-79b7.local` (ARM64, Blackwell GPU) | GPU training, tests, and Triton deploy |

Runners are managed in [miramar-platform-gcp](https://github.com/miramar-labs-org/miramar-platform-gcp) (`mlabs-runner/` for the Docker image, `scripts/gha/launch-runner.sh` to register). The DGX runner mounts the host Docker socket — `--gpus all` works because the Docker daemon on the DGX host has GPU access. Both containers mount `$HOME/.cache/huggingface` from the DGX host so model weights and the IMDB dataset are downloaded once and reused.

## Triton Inference

After deployment, test via port-forward:

```bash
kubectl --context minikube port-forward -n mlops-torch-triton-dgx-pipeline svc/triton 8000:8000

# Health
curl localhost:8000/v2/health/ready

# Inference (tokenize your text to input_ids + attention_mask, pad to length 128)
curl -X POST localhost:8000/v2/models/text_classifier/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "input_ids",      "shape": [1, 128], "datatype": "INT64", "data": [101, ...]},
      {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64", "data": [1, ...]}
    ]
  }'
```

Logits output: index 0 = negative, index 1 = positive. Apply softmax for probabilities.

## Deploy Notes

- `kubectl apply` includes the `kind: Namespace` document — minikube handles re-applying an existing namespace cleanly.
- The Triton serving image has no `nvidia.com/gpu` resource request in `k8s/triton.yaml` — it runs CPU-only unless a GPU device plugin is configured in minikube.
- Image tags: both `:latest` and the commit SHA are pushed to Docker Hub; the SHA tag ensures the deployed image always matches the trained model from that exact run.
