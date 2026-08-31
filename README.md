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
│   ├── capacity/               # vLLM capacity testing & saturation profiling
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
        ├── capacity-test.yaml   # In-cluster capacity & saturation testing
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

### Step 2: Deploy / Spin Up a Model via Kustomize

To start your model server in the morning (or switch models), run `kubectl apply` for your target overlay. Running this command will spin up the GPU pod:

> [!NOTE]
> **KEDA 1-Hour Auto-Shutdown Safeguard**: KEDA monitors query activity. If the inference pod remains completely idle for **1 hour** (3,600s), KEDA will automatically scale it down to `0` replicas to prevent unnecessary GPU compute charges. Running `kubectl apply -k k8s/overlays/<model>/` in the morning will always spin it back up to 1 replica.

#### A. Gemma 3 Models (1x L4 GPU)
```bash
# Gemma 3 1B:
kubectl apply -k k8s/overlays/gemma-3-1b/

# Gemma 3 4B (Instance 1):
kubectl apply -k k8s/overlays/gemma-3-4b/

# Gemma 3 4B (Instance 2 - Isolated Endpoint):
kubectl apply -k k8s/overlays/gemma-3-4b-2/

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

#### E. Qwen 3.8 27B (4x L4 GPUs or 1x A100 GPU)
```bash
# 4x L4 GPUs (Unquantized BF16 — 96GB VRAM):
kubectl apply -k k8s/overlays/qwen-3.8-27b-4xl4/

# 1x A100 80GB GPU (Unquantized BF16):
kubectl apply -k k8s/overlays/qwen-3.8-27b-a100/
```

#### F. Custom Fine-Tuned Model / LoRA (Streamed via GCS FUSE)
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

## Capacity & Saturation Testing Guide

Measure serving limits, TTFT, TPOT, KV cache memory occupancy, and find the optimal concurrency sweet spot ($C^*$) before running benchmarks:

### Run an In-Cluster Capacity Sweep (Batch / GSM8K / MMLU):
```bash
# 1. Update benchmark ConfigMap
kubectl create configmap capacity-benchmark-scripts \
  --from-file=benchmark_capacity.py=benchmarks/capacity/benchmark_capacity.py \
  --from-file=gsm8k.jsonl=benchmarks/capacity/datasets/gsm8k.jsonl \
  --from-file=mmlu.jsonl=benchmarks/capacity/datasets/mmlu.jsonl \
  -n inference --dry-run=client -o yaml | kubectl apply -f -

# 2. Trigger capacity test job
kubectl apply -f k8s/jobs/capacity-test.yaml

# 3. Stream logs
kubectl logs -f job/gemma3-capacity-test -c capacity-runner -n inference
```

See [benchmarks/capacity/README.md](benchmarks/capacity/README.md) for full documentation, workload presets, and empirical results for Gemma 3 and Qwen 3.8.

### Empirical Benchmark Results: Qwen 3.8 27B (1x A100 80GB vs. 4x L4 24GB)

#### A. 1x NVIDIA A100 80GB SXM4 (`a2-ultragpu-1g` — Single GPU, TP=1)

| Concurrency Tier | Output Tok/s | Req/s | TTFT p50 (ms) | TPOT p50 (ms) | Latency p95 (s) | Peak KV % | Status / Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 219.04 | 0.58 | **17.65 ms** | 33.91 ms | 3.12 s | 0.1% | Single Stream Speed |
| **2** | 379.29 | 1.11 | **20.85 ms** | 34.02 ms | 3.19 s | 0.2% | Linear Scaling |
| **4** | 588.66 | 1.98 | **27.24 ms** | 34.31 ms | 3.45 s | 0.3% | Linear Scaling |
| **8** | 798.11 | 3.44 | **36.43 ms** | 34.82 ms | 3.75 s | 0.7% | High Efficiency |
| **16 (Peak)** | **923.36** | **5.18** | **58.74 ms** | **36.19 ms** | **4.31 s** | **1.4%** | ★ **Optimal Saturation ($C^*=16$)** |
| **32** | 812.45 | 6.13 | 122.95 ms | 40.54 ms | 7.15 s | 3.4% | Saturation Plateau (-12.0%) |
| **48** | 672.02 | 6.12 | 204.42 ms | 45.18 ms | 10.74 s | 5.4% | Queue Throttling (-27.2%) |
| **64** | 541.19 | 6.09 | 308.83 ms | 51.62 ms | 14.52 s | 7.6% | Batch Contention (-41.4%) |

#### B. 4x NVIDIA L4 24GB (`g2-standard-48` — Multi-GPU, TP=4)

| Concurrency Tier | Output Tok/s | Total Tok/s | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | Status / Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 149.95 | 467.28 | 29.10 ms | 35.80 ms | 6.65 ms | 7.10 ms | Single Stream Speed |
| **2** | 253.67 | 789.98 | 41.00 ms | 57.80 ms | 7.80 ms | 8.30 ms | Linear Scaling |
| **4** | 380.69 | 1,208.91 | 72.00 ms | 110.20 ms | 9.90 ms | 11.00 ms | Linear Scaling |
| **8** | 527.42 | 1,638.54 | 138.20 ms | 211.50 ms | 13.90 ms | 15.80 ms | High Efficiency |
| **16 (Peak)** | **556.32** | **1,730.30** | 271.10 ms | **405.20 ms** | 21.70 ms | **24.90 ms** | ★ **Optimal Saturation ($C^*=16$)** |
| **32** | 452.86 | 1,408.85 | 499.80 ms | 752.10 ms | 34.10 ms | 39.20 ms | Degrading (-18.6%) |
| **48** | 362.34 | 1,126.99 | 775.20 ms | 1,150.40 ms | 49.10 ms | 56.40 ms | Queue Bottleneck (-34.8%) |
| **64** | 309.83 | 963.54 | 1,038.40 ms | 1,540.80 ms | 64.20 ms | 73.80 ms | Severe Throttling (-44.3%) |

#### C. Architectural Comparison: 1x A100 80GB vs. 4x L4 24GB

| Characteristic / Metric | 1x A100 80GB SXM4 (Single GPU) | 4x L4 24GB (TP=4 PCIe) | Architectural Trade-Off |
| :--- | :---: | :---: | :--- |
| **Memory Bandwidth** | **2,039 GB/s (HBM2e)** | 1,200 GB/s ($4 \times 300$ GB/s GDDR6) | **1.7x higher bandwidth** on A100 accelerates memory-bound decoding |
| **Tensor Compute (BF16)** | 312 TFLOPS (dense) | **484 TFLOPS** ($4 \times 121$ TFLOPS) | 4x L4 has more raw FLOPS, but suffers TP communication overhead |
| **Peak Throughput ($C=16$)** | **923.36 output tok/s** | 556.32 output tok/s | **+66.0% higher generation throughput** on A100 |
| **Single Stream TTFT ($C=1$)** | **17.65 ms** | 29.10 ms | **39.3% faster prefill** on A100 |
| **Saturation TTFT ($C=16$)** | **58.74 ms** | 271.10 ms | **78.3% lower TTFT** (no PCIe `all-reduce` contention) |
| **KV Cache VRAM Headroom** | 17.74 GiB (11,352 blocks) | **44.18 GiB** (28,272 blocks) | 4x L4 provides **2.5x larger KV cache** for ultra-long context |
| **Operational Simplicity** | Monolithic (TP=1, no NCCL sync) | Distributed (TP=4 over PCIe) | A100 has simpler failure domain & zero cross-GPU latency jitter |

---


## Benchmarking Guide ($\tau^2$-bench)

Execute in-cluster evaluation jobs that query the internal vLLM ClusterIP service and stream trajectories and score cards directly to Cloud Storage.

### Run a Throughput Saturation Benchmark Job ($C^*=96$):
```bash
kubectl apply -f k8s/jobs/tau-bench-saturation-job.yaml
```

### Stream Saturation Job Logs:
```bash
kubectl logs -f job/tau-bench-saturation-run -n inference
```

### Run a Standard Benchmark Job:
```bash
kubectl apply -f k8s/jobs/benchmark-tau2.yaml
```

### Stream Benchmark Logs:
```bash
kubectl logs -f job/tau2-bench-run-qwen7b -n inference
```

### Inspect Output in Cloud Storage:
```bash
gcloud storage ls gs://efficient-inference-506713-models/benchmarks/capacity/tau_bench_saturation/
```

---

## Mitigation & Optimization Experiments (Experiments 1–4)

To measure and compare latency, throughput, and accuracy across formatting mitigation strategies:

### Experiment 1: Strict System Instructions & Formatting Prompts
Enforces raw JSON formatting rules, negative constraints against markdown codeblocks, and explicit 1024 token limit + stop tokens.
```bash
kubectl apply -f k8s/jobs/exp1-strict-prompt-job.yaml
```

### Experiment 2: Error Feedback & Self-Correction Loop
Intercepts JSON parsing errors, feeds exact `JSONDecodeError` messages back to the model, and allows a 1-turn self-correction retry.
```bash
kubectl apply -f k8s/jobs/exp2-self-correct-job.yaml
```

### Experiment 3: Combined Strict Prompting + Self-Correction Loop
Combines Experiment 1 prompt constraints + max token limits WITH Experiment 2 error feedback retries.
```bash
kubectl apply -f k8s/jobs/exp3-combined-job.yaml
```

### Experiment 4: Supervised Fine-Tuning (SFT) & LoRA Serving
1. **Prepare SFT Dataset**: `python3 benchmarks/capacity/prepare_sft_dataset.py`
2. **Train LoRA Adapter**: `python3 benchmarks/capacity/train_sft_lora.py`
3. **Deploy SFT LoRA Model Overlay**: `kubectl apply -k k8s/overlays/gemma-3-4b-sft/`
4. **Evaluate SFT Model**: `kubectl apply -f k8s/jobs/exp4-sft-job.yaml`



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
