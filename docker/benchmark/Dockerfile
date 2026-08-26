FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy tau2-bench benchmark repository
COPY benchmarks/tau2-bench /app/tau2-bench
WORKDIR /app/tau2-bench

RUN uv pip install --system -e .

COPY docker/benchmark/run_bench.sh /app/run_bench.sh
RUN chmod +x /app/run_bench.sh

ENTRYPOINT ["/app/run_bench.sh"]
