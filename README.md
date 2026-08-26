# Efficient LLM Inference, Fine-Tuning & Benchmarking on GKE

A cost-optimized infrastructure on Google Kubernetes Engine (GKE) and Google Cloud Platform (GCP) for self-hosting sub-30B LLMs (Qwen 2.5, Gemma 2, DeepSeek-R1 Distill), running post-training / fine-tuning experiments, and executing agent benchmarks ($\tau^2$-bench / $\tau^3$-bench) strictly within a **\$2,000/month budget**.

---

## Architecture Highlights

- **Accelerators**: NVIDIA L4 (24GB VRAM) on `g2-standard-4` (1x GPU) and `g2-standard-24` (2x GPU) with Spot instance support for batch jobs.
- **Weight Streaming**: Zero-copy streaming from Cloud Storage (`gs://efficient-inference-506713-models`) via the **GCS FUSE CSI Driver**.
- **Declarative Deployments**: Managed via **Kustomize** overlays for easy model switching without YAML duplication.
- **Cost Efficiency**: GPU node pools auto-scale from **0 replicas**, spinning up on demand in ~90s and releasing GPU VMs when idle.

---

## Prerequisites & Cluster Access

### 1. Install `kubectl` & GKE Auth Plugin (on Cloudtop / gLinux)
```bash
sudo apt-get update && sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin
```

### 2. Connect to the GKE Cluster
```bash
gcloud container clusters get-credentials efficient-inference-cluster \
  --zone=us-central1-a \
  --project=efficient-inference-506713
```

Verify connectivity:
```bash
kubectl get nodes
```

---

## Repository Structure

```
.
├── benchmarks/
│   └── tau2-bench/             # tau2 / tau3 benchmark framework
├── docker/
│   ├── vllm/
│   │   ├── Dockerfile          # PyTorch 2.4 + CUDA 12.4 + vLLM + FlashInfer
│   │   └── entrypoint.sh       # Dynamic model/LoRA entrypoint
│   └── benchmark/
│       ├── Dockerfile          # Containerized benchmark runner
│       └── run_bench.sh        # Automated evaluation script
└── k8s/
    ├── 00-namespace-and-sa.yaml # Namespace `inference`, KSA, and secrets
    ├── base/                    # Base vLLM Deployment, Service, and KEDA ScaledObject
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   ├── scaledobject.yaml    # Auto-targets active model deployment
    │   ├── kustomizeconfig.yaml # Dynamic nameReference resolution
    │   └── kustomization.yaml
    ├── overlays/                # Model-specific overlays
    │   ├── qwen-7b/             # Qwen2.5-7B-Instruct (1x L4)
    │   ├── gemma-2-9b/          # Gemma-2-9B-It (1x L4)
    │   ├── gemma-3-1b/          # Gemma-3-1B-It (1x L4)
    │   ├── gemma-3-4b/          # Gemma-3-4B-It (1x L4)
    │   ├── gemma-3-12b-fp8/     # Gemma-3-12B-It FP8 Quantized (1x L4)
    │   ├── deepseek-r1-7b/      # DeepSeek-R1-Distill-Qwen-7B (1x L4)
    │   ├── deepseek-r1-14b/     # DeepSeek-R1-Distill-Qwen-14B (2x L4)
    │   └── custom-lora/         # Custom LoRA adapter streamed via GCS FUSE
    └── jobs/                    # On-demand batch evaluation & training jobs
        └── benchmark-tau2.yaml  # tau2-bench evaluation job
```

---

## Model Serving Deployment Guide

### Step 1: Initialize Namespace, KEDA Operator & Secrets (One-time)
```bash
# 1. Apply base namespace, KSA, and secrets
kubectl apply -f k8s/00-namespace-and-sa.yaml

# 2. Install KEDA operator and CRDs for scale-to-zero autoscaling
kubectl apply --server-side -f https://github.com/kedacore/keda/releases/download/v2.14.0/keda-2.14.0.yaml
```

*(Optional: If deploying gated models like Gemma, inject your Hugging Face token:)*
```bash
kubectl create secret generic hf-secret \
  --from-literal=token="hf_YOUR_TOKEN" \
  -n inference --dry-run=client -o yaml | kubectl apply -f -
```

---

### Step 2: Deploy a Model via Kustomize

Deploy any model with a single command. The GKE Cluster Autoscaler will provision the required L4 GPU node on demand (~90s), and KEDA will automatically attach a `ScaledObject` to scale down to 0 when idle:

#### A. Gemma 3 Models (1x L4 GPU)
```bash
# Gemma 3 1B:
kubectl apply -k k8s/overlays/gemma-3-1b/

# Gemma 3 4B:
kubectl apply -k k8s/overlays/gemma-3-4b/

# Gemma 3 12B FP8 (Quantized):
kubectl apply -k k8s/overlays/gemma-3-12b-fp8/
```

#### B. Qwen 2.5 7B Instruct (1x L4 GPU)
```bash
kubectl apply -k k8s/overlays/qwen-7b/
```

#### C. DeepSeek R1 Distill Qwen 7B (1x L4 GPU)
```bash
kubectl apply -k k8s/overlays/deepseek-r1-7b/
```

#### D. DeepSeek R1 Distill Qwen 14B (2x L4 GPUs)
```bash
kubectl apply -k k8s/overlays/deepseek-r1-14b/
```

#### E. Custom Fine-Tuned Model / LoRA (Streamed via GCS FUSE)
```bash
kubectl apply -k k8s/overlays/custom-lora/
```

---

### Step 3: Verify Status & Send Test Requests

#### Check Pod & GPU Node Status:
```bash
kubectl get pods -n inference -w
```

#### Port-Forward the Service Locally:
```bash
kubectl port-forward svc/vllm-service-qwen-7b 8000:8000 -n inference
```

#### Send an OpenAI-Compatible Chat Request:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain KV caching in two sentences."}
    ]
  }'
```

---

## Benchmarking Guide ($\tau^2$-bench)

Execute in-cluster evaluation jobs that query the internal vLLM ClusterIP service and stream trajectories and score cards directly to Cloud Storage.

### Run a Benchmark Job:
```bash
kubectl apply -f k8s/jobs/benchmark-tau2.yaml
```

### Stream Benchmark Logs:
```bash
kubectl logs -f job/tau2-bench-run-qwen7b -n inference
```

### Inspect Output in Cloud Storage:
```bash
gcloud storage ls gs://efficient-inference-506713-models/benchmarks/tau2-bench/
```

---

## Cost Optimization & Resource Teardown

To keep monthly compute costs minimal:

### 1. Scale Model to 0 (Preserves Service, Deprovisions GPU Node)
```bash
kubectl scale deployment vllm-server-qwen-7b --replicas=0 -n inference
```
*(GKE's cluster autoscaler will terminate the GPU VM after ~5 minutes of zero active pods).*

### 2. Complete Teardown of a Model
```bash
kubectl delete -k k8s/overlays/qwen-7b/
```
