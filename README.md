# mlops-torch-triton-dgx-pipeline

GPU ML training pipeline: fine-tune DistilBERT for text classification on a DGX Station, track experiments with MLflow, serve the model on minikube via Triton Inference Server.

[![ML Train Test](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-train-test.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-train-test.yaml)
[![ML Train](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-train.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-train.yaml)
[![ML Build Push](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-build-push.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-build-push.yaml)
[![ML Deploy](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-deploy.yaml/badge.svg)](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions/workflows/ml-deploy.yaml)

## Links

- **[GitHub Actions](https://github.com/miramar-labs-org/mlops-torch-triton-dgx-pipeline/actions)** — workflow run history
- **[MLflow UI](http://localhost:5000)** — experiment tracking on DGX; requires SSH tunnel: `ssh -L 5000:localhost:5000 aaron@spark-79b7.local`
- **[minikube Dashboard](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/)** — requires SSH tunnel: `ssh -L 8001:localhost:8001 aaron@spark-79b7.local` (proxy runs on DGX: `nohup kubectl --context minikube proxy --port=8001 --address=127.0.0.1 > ~/kubectl-proxy.log 2>&1 &`)

## Local Development

```bash
# Create a pyenv virtualenv for this workspace (once)
pyenv virtualenv 3.13.0 pyTriton
pyenv local pyTriton

# Install dependencies (CPU torch — enough for tests and linting)
pip install -r ml/requirements.txt
```

VS Code will use the `pyTriton` interpreter automatically via `.python-version`.

## Pipeline

```
ML Train Test — workflow_dispatch or push to ml/{train.py,test_train.py,Dockerfile.train} (dgx, ARM64)
  ├── pytest → test_train.py  (aborts chain on failure; deps pre-installed in mlabs-runner)
  └── docker build → ml-trainer image

ML Train — triggered by ML Train Test success (dgx, ARM64, GPU)
  ├── docker build → ml-trainer image
  ├── docker run --gpus all → DistilBERT fine-tune on IMDB
  ├── log metrics → MLflow (localhost:5000, via --network host)
  ├── export → model.onnx
  └── upload artifact → onnx-model

ML Build Push — triggered by ML Train success (dgx, ARM64)
  ├── download artifact → model.onnx
  ├── docker build → Triton serving image (model baked in)
  └── push → Docker Hub (latest + SHA tag)

ML Deploy — triggered by ML Build Push success (dgx, ARM64)
  └── kubectl apply → minikube namespace mlops-torch-triton-dgx-pipeline
```

## Workflows

| Workflow | File | Runner | Trigger |
|---|---|---|---|
| **ML Train Test** | `ml-train-test.yaml` | `dgx` | Push to `ml/train.py`, `test_train.py`, or `Dockerfile.train`; or manual |
| **ML Train** | `ml-train.yaml` | `dgx` | Auto on ML Train Test success; or manual |
| **ML Build Push** | `ml-build-push.yaml` | `dgx` | Auto on ML Train success; or manual with `run_id` |
| **ML Deploy** | `ml-deploy.yaml` | `dgx` | Auto on ML Build Push success; or manual with `image_tag` (git SHA) |

### ML Train inputs (manual dispatch only)

| Input | Default | Description |
|---|---|---|
| `epochs` | `3` | Number of training epochs |
| `experiment` | `text-classifier` | MLflow experiment name |

The model artifact (`onnx-model`) passes between workflows via GitHub Actions artifact storage. The commit SHA is used as the Docker Hub image tag throughout — `latest` and the SHA tag always correspond to the same trained model.

## Model

| Property | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Binary sentiment classification (IMDB) |
| Dataset | HuggingFace `datasets` — `imdb` (25k train / 25k test) |
| Max sequence length | 128 tokens |
| Export format | ONNX (opset 14) |
| Inference backend | Triton ONNX Runtime |

## Runners

One self-hosted runner is required for all three workflows. The runner image (`mlabs-runner`) and launch scripts live in [miramar-platform-gcp](https://github.com/miramar-labs-org/miramar-platform-gcp).

| Runner | Label | Host | Used for |
|---|---|---|---|
| NVIDIA DGX Spark 128GB | `dgx` | `spark-79b7.local` (ARM64, Blackwell GPU) | GPU training, tests, and Triton deploy |

The DGX runner mounts the host Docker socket — `--gpus all` works because the Docker daemon on the DGX host has GPU access. Training and test containers mount `$HOME/.cache/huggingface` from the DGX host so the `distilbert-base-uncased` tokenizer and IMDB dataset (~200 MB) are downloaded once and reused across runs.

## GitHub Secrets and Variables

| Name | Scope | Type | Description |
|---|---|---|---|
| `DOCKERHUB_USERNAME` | org | Secret | Docker Hub username for pushing the Triton serving image |
| `DOCKERHUB_TOKEN` | org | Secret | Docker Hub access token |
| `DGX_MINIKUBE_KUBECONFIG` | org | Secret | Base64-encoded kubeconfig for the minikube cluster on the DGX |

`MLFLOW_TRACKING_URI` is set via a GitHub org variable. The training container runs with `--network host` so it reaches MLflow on the DGX loopback directly.

## Triton Inference

After deployment, access via port-forward on the DGX:

```bash
kubectl --context minikube port-forward -n mlops-torch-triton-dgx-pipeline svc/triton 8000:8000

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

Logits: index 0 = negative, index 1 = positive. Apply softmax for probabilities.

Triton also exposes gRPC on port 8001 and Prometheus metrics on port 8002.

## minikube Dashboard

The minikube cluster on the DGX has the dashboard enabled. Start the proxy on the DGX once (survives across sessions):

```bash
nohup kubectl --context minikube proxy --port=8001 --address=127.0.0.1 > ~/kubectl-proxy.log 2>&1 &
```

Then SSH tunnel from your laptop and open the link above:

```bash
ssh -L 8001:localhost:8001 aaron@spark-79b7.local
```

## Repository Structure

```
.github/workflows/
  ml-train-test.yaml       # Build training image and run pytest (entry point)
  ml-train.yaml            # GPU training on DGX — exports model.onnx as artifact
  ml-build-push.yaml       # Build Triton serving image and push to Docker Hub
  ml-deploy.yaml           # Deploy to minikube from Docker Hub image
ml/
  train.py              # DistilBERT fine-tune + ONNX export + MLflow logging
  test_train.py         # Unit tests for tokenize_batch, evaluate, ONNX export
  requirements.txt      # Local dev dependencies (CPU torch + pytest)
  Dockerfile.train      # GPU training image (nvcr.io/nvidia/pytorch:25.03-py3)
  Dockerfile.serve      # Triton serving image (model.onnx baked in at build time)
  triton_config.pbtxt   # Triton model config (ONNX Runtime backend)
  output/               # Generated at runtime — model.onnx (gitignored)
k8s/
  triton.yaml           # Namespace + Deployment + Service for Triton on minikube
```
