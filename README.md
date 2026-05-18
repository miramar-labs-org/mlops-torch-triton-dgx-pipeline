# mlops-pipeline

GPU ML training pipeline: fine-tune DistilBERT for text classification on a DGX Station, track experiments with MLflow, serve the model on GKE via Triton Inference Server.

[![ML Train](https://github.com/miramar-labs/mlops-pipeline/actions/workflows/ml-train.yaml/badge.svg)](https://github.com/miramar-labs/mlops-pipeline/actions/workflows/ml-train.yaml)

## Links

- **[GCP Artifact Registry](https://console.cloud.google.com/artifacts/docker/miramar-platform/us-west1/apps?project=miramar-platform)** — `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier`
- **[GKE Workloads](https://console.cloud.google.com/kubernetes/workload/overview?project=miramar-platform)** — `triton` deployment in namespace `ml-serving` on `miramar-shared-gke`
- **[GitHub Actions](https://github.com/miramar-labs/mlops-pipeline/actions)** — workflow run history

## Pipeline

```
workflow_dispatch
  └── train job (dgx-spark)
        ├── docker run --gpus all → DistilBERT fine-tune on IMDB
        ├── log metrics → MLflow (host.docker.internal:5000)
        ├── export → model.onnx
        ├── docker build → Triton serving image (model baked in)
        └── push → GAR
  └── deploy-triton job (needs: train)
        └── kubectl apply → GKE namespace ml-serving
```

## Workflow

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| **ML Train** | `ml-train.yaml` | manual | Fine-tunes DistilBERT on IMDB, logs to MLflow, exports ONNX, builds Triton image, deploys to GKE |

**Inputs:**

| Input | Default | Description |
|---|---|---|
| `epochs` | `3` | Number of training epochs |
| `experiment` | `text-classifier` | MLflow experiment name |

## Model

| Property | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Binary sentiment classification (IMDB) |
| Dataset | HuggingFace `datasets` — `imdb` (25k train / 25k test) |
| Max sequence length | 128 tokens |
| Export format | ONNX (opset 14) |
| Inference backend | Triton ONNX Runtime |

## MLflow Setup (one-time on DGX host)

MLflow runs on the DGX and is accessible to training containers via `host.docker.internal:5000`.

```bash
pip install mlflow
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Access the UI at `http://localhost:5000` (or via SSH port-forward from another machine).

## GCP / GKE

| Resource | Value |
|---|---|
| Project | `miramar-platform` |
| Cluster | `miramar-shared-gke` (`us-west1-a`) |
| Namespace | `ml-serving` |
| Artifact Registry | `us-west1-docker.pkg.dev/miramar-platform/apps/triton-text-classifier` |
| Auth | Workload Identity Federation — no long-lived keys |

## Runner

The workflow runs on the DGX self-hosted runner (`dgx-spark`). The runner image (`ghcr.io/miramar-labs/github-runner:latest`) and launch scripts live in [github-actions-hello](https://github.com/miramar-labs/github-actions-hello). This repo includes a copy of `runner/` and `scripts/` for convenience.

**Launch the runner for this repo** (get a fresh token from Settings → Actions → Runners → New self-hosted runner):
```bash
./runner/launch.sh TOKEN https://github.com/miramar-labs/mlops-pipeline
```

`launch.sh` auto-detects architecture, pulls the latest image, and registers against the supplied repo URL.

## GitHub Secrets and Variables Required

| Secret/Variable | Type | Description |
|---|---|---|
| `WIF_PROVIDER` | Secret | Workload Identity Federation provider |
| `GCP_SERVICE_ACCOUNT` | Secret | GCP service account email for WIF |
| `MLFLOW_TRACKING_URI` | Variable | MLflow tracking URI (e.g. `http://host.docker.internal:5000`) |
| `RUNNER_LABELS` | Variable | Runner label (default: `dgx-spark`) |

## Triton Inference

After deployment, access via port-forward:

```bash
kubectl port-forward -n ml-serving svc/triton 8000:8000

# Health check
curl localhost:8000/v2/health/ready

# Inference (input_ids and attention_mask as INT64 tensors, length 128)
curl -X POST localhost:8000/v2/models/text_classifier/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "input_ids",      "shape": [1, 128], "datatype": "INT64", "data": [101, ...]},
      {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64", "data": [1, ...]}
    ]
  }'
```

Triton also exposes gRPC on port 8001 and Prometheus metrics on port 8002.

## Repository Structure

```
.github/workflows/
  ml-train.yaml         # Training + deploy workflow
ml/
  train.py              # DistilBERT fine-tune + ONNX export + MLflow logging
  Dockerfile.train      # GPU training image (pytorch/pytorch:2.3.0-cuda12.1)
  Dockerfile.serve      # Triton serving image (model baked in)
  triton_config.pbtxt   # Triton model config (ONNX Runtime backend)
  output/               # Generated at runtime — model.onnx (gitignored)
k8s/
  triton.yaml           # Namespace + Deployment + Service for Triton on GKE
```
