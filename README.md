# RMLTP — Distributed Machine Learning Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ZeroMQ](https://img.shields.io/badge/ZeroMQ-ROUTER%2FDEALER-brightgreen.svg)](https://zeromq.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, from-scratch distributed ML training platform combining ideas from **Kubernetes** (pod orchestration, health checks, heartbeats), **Ray** (dynamic task scheduling, actor model), and **PyTorch Distributed** (Data Parallelism, Pipeline Parallelism, AllReduce, Parameter Server).

---

## 📚 Table of Contents
1. [Core Features](#-core-features)
2. [Architecture Overview](#-architecture-overview)
3. [Quick Start (Local Multi-Terminal)](#-quick-start-local-multi-terminal)
4. [Live Resilience Demo (Process Kill & Recovery)](#-live-resilience-demo-process-kill--recovery)
5. [Containerization (Docker & Docker Compose)](#-containerization-docker--docker-compose)
6. [Free Cloud Hosting Deployment (Hugging Face Spaces / Render)](#-free-cloud-hosting-deployment)
7. [Study Guide & Documentation](#-study-guide--documentation)
8. [Testing & Verification](#-testing--verification)

---

## 🚀 Core Features

- **True Multi-Process Networking**: Autonomous worker processes communicating over ZeroMQ TCP sockets using non-blocking `ROUTER`/`DEALER` patterns.
- **REST Job Submission API**: Submit jobs with custom training code, dataset location, resource specifications, and worker counts (`/api/jobs/submit`).
- **Distributed Scheduling**: Priority-based scheduling queue with locality-aware worker placement.
- **Worker Registration & Heartbeats**: Autonomous worker daemons report CPU/GPU hardware specs and stream 2-second heartbeats over TCP.
- **Fault-Tolerant Self-Healing**: Live process crash detection with automatic task rescue and reassignment.
- **Distributed ML Primitives**:
  - **Data Parallelism**: `DataParallelExecutor` with dataset sharding.
  - **Pipeline Parallelism**: `PipelineParallelStage` with backward activation propagation.
  - **Synchronization**: Ring-style `AllReduceSynchronizer` and central `ParameterServer` (synchronous and asynchronous with staleness damping $\frac{1}{1 + \tau}$).
- **Advanced Capabilities**:
  - **Gradient Compression**: Top-K gradient sparsification and Gzip serialization.
  - **Distributed Optimizer**: Drop-in PyTorch optimizer wrapper with automatic gradient reduction on `.step()`.
  - **Fault-Tolerant AllReduce**: Collective reduction that automatically excludes dead nodes without barrier deadlocks.
  - **Elastic Scaling**: Dynamic addition and removal of worker nodes with dataset re-sharding.
- **Observability**: Prometheus metrics exporter (`9090`) and interactive Dark-Themed Web Dashboard (`dashboard.py`).

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    Client["Client / User"] -->|POST /api/jobs/submit| JobAPI["Job Submission API\n(Port 5000 / 7860)"]
    JobAPI --> Controller["Master Controller\n(Port 8000)"]
    
    subgraph Controller_Subsystem["Controller Core"]
        Scheduler["Task Scheduler\n(Priority Queue)"]
        HealthMonitor["Worker Health Monitor\n(Heartbeat Sweeper)"]
        CkptMgr["Checkpoint Manager\n(Atomic Writes)"]
        Metrics["Prometheus Exporter\n(Port 9090)"]
    end
    
    Controller <-->|ZeroMQ ROUTER tcp://0.0.0.0:8000| CommBus((ZMQ Message Bus))
    
    CommBus <-->|ZeroMQ DEALER| Worker1["Worker Node 1\n(PID 1, ModelRunner)"]
    CommBus <-->|ZeroMQ DEALER| Worker2["Worker Node 2\n(PID 2, ModelRunner)"]
    CommBus <-->|ZeroMQ DEALER| WorkerN["Worker Node N\n(PID N, ModelRunner)"]
```

---

## 💻 Quick Start (Local Multi-Terminal)

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Multi-Terminal Demonstration
Open 4 separate terminal windows:

- **Terminal 1: Start Controller Orchestrator**
  ```bash
  python src/controller/main.py --host 127.0.0.1 --port 8000 --mode serve
  ```

- **Terminal 2: Launch Worker Node 1**
  ```bash
  python src/node/main.py --id worker-1 --port 8001 --controller 127.0.0.1:8000
  ```

- **Terminal 3: Launch Worker Node 2**
  ```bash
  python src/node/main.py --id worker-2 --port 8002 --controller 127.0.0.1:8000
  ```

- **Terminal 4: Start Interactive Web Dashboard**
  ```bash
  python dashboard.py --port 5000
  ```
  Open [http://localhost:5000](http://localhost:5000) in your browser to view live telemetry, trigger distributed training steps, and simulate node failures!

---

## ⚡ Live Resilience Demo (Process Kill & Recovery)

Run the automated multi-process resilience demo script:
```bash
python demo_cluster.py
```

This script:
1. Spawns the Controller and two autonomous Workers as real OS subprocesses.
2. Waits for registration handshakes and heartbeats.
3. Forcefully sends `SIGKILL` to Worker 2.
4. Shows the Controller detecting the node crash, marking it dead, and seamlessly failing over tasks to Worker 1.

---

## 🐳 Containerization (Docker & Docker Compose)

### Single-Container Multi-Process Supervisor
```bash
# Build the Docker image
docker build -t rmltp-platform:latest .

# Run container (Dashboard on port 7860)
docker run -p 7860:7860 -p 8000:8000 -p 9090:9090 rmltp-platform:latest
```

### Multi-Container Cluster with Docker Compose
```bash
docker-compose up --build
```
This starts isolated containers for `controller`, `worker-1`, `worker-2`, and `dashboard` connected across an internal bridge network (`rmltp-net`).

---

## ☁️ Free Cloud Hosting Deployment

### Recommended: Hugging Face Spaces (Docker SDK)
Hugging Face Spaces provides **16GB RAM, 2 vCPUs completely FREE** without requiring a credit card:

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **New Space** -> Choose **Docker** as Space SDK -> Select **Blank**.
3. Clone the repo and push your code:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space master:main
   ```
4. Hugging Face Spaces automatically builds the `Dockerfile`, invokes `start_container.sh`, and serves the live Web Dashboard on port `7860`!

### Alternative: Render Free Web Service
1. Connect your GitHub repository to [Render](https://render.com/).
2. Select **Web Service** -> Environment **Docker**.
3. Set environment variable `PORT=10000`.
4. Deploy for free.

---

## 📖 Study Guide & Documentation

- [learnings.md](learnings.md): **Comprehensive 10-Chapter Masterclass** covering distributed computing theory, Data/Pipeline Parallelism, SGD math, AllReduce vs Parameter Server, ZeroMQ topologies, fault tolerance, and interview cheat sheets.
- [tasks.md](tasks.md): **30-Point Implementation Checklist** tracking every requirement.
- [specs/proposals/001-real-multiprocess-distributed-platform.md](specs/proposals/001-real-multiprocess-distributed-platform.md): Formal OpenSpec architectural proposal.

---

## 🧪 Testing & Verification

Run the entire 117-test verification suite (unit, integration, and benchmarks):
```bash
pytest -v
```

Run benchmarks for communication overhead and tensor serialization:
```bash
python benchmarks/benchmark_distributed.py
```

---

## 📄 License
MIT License.
