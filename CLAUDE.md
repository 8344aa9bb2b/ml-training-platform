# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RMLTP (Ray-based ML Training Platform): A distributed ML training platform inspired by Kubernetes + Ray + PyTorch Distributed, providing:
- **Controller**: Orchestrates training jobs, manages resources, handles scheduling
- **Scheduler**: Distributes work across nodes with support for mixed precision, pipeline parallelism, and automatic scaling
- **Node Agents**: Lightweight workers that execute training tasks, report metrics, and handle failures
- **Model Runner**: Handles model-specific training loops, hyperparameter tuning, and experiment tracking
- **Network**: Communication layer for parameter sharing, synchronization, and data distribution

## Core Technologies

- **Python 3.10+** - Primary language
- **PyTorch** - Deep learning framework
- **Ray** - Distributed computing framework (optional dependency, may substitute with custom implementation)
- **ZeroMQ / gRPC** - Communication layer between nodes
- **Prometheus** - Metrics and monitoring
- **Docker** - Containerization (optional, for production deployment)

## Directory Structure

```
.
├── src/              # Core application code
│   ├── controller/   # Master orchestrator, job submission, resource management
│   ├── scheduler/    # Work distribution, topology orchestration
│   ├── node/         # Worker agents, training execution
│   ├── model/        # Training loops, optimization utilities
│   └── network/      # Communication protocols
├── core/             # Shared core utilities, config management
├── config/           # Configuration files, environment templates
├── tests/            # Unit and integration tests
├── benchmarks/       # Performance benchmarks
├── logs/             # Runtime logs (outside version control)
└── docs/             # Architecture docs, diagrams, API references
```

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or with Ray support
pip install -r requirements[ray].txt
```

### Running the Platform
```bash
# Start controller (master)
python src/controller/main.py --config config/controller.yaml

# Start scheduler (standalone component)
python src/scheduler/main.py --config config/scheduler.yaml

# Start a worker node
python src/node/worker.py -- nodes.yaml

# Run single training job from controller
python src/controller/main.py --mode job --job-config config/job.yaml
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_scheduler.py

# Run integration tests
pytest tests/integration/ -v

# Run with mock distributed environment
pytest tests/ -m distributed
```

### Monitoring
```bash
# View controller logs
tail -f logs/controller.log

# Check node status via REST API (port 8000)
curl http://localhost:8000/nodes

# Prometheus metrics endpoint (port 9090)
curl http://localhost:9090/metrics
```

### Development
```bash
# Lint code
flake8 src/ tests/

# Format code
black src/ tests/

# Type checking
mypy src/

# Security scanning
safety check

# Dependency updates
pip-compile requirements.in
```

## Architecture Notes

### High-Level Architecture

The platform operates in a master-worker model:
1. **Controller** acts as the central orchestrator - accepts job submissions, allocates resources, assigns tasks to scheduler/nodes
2. **Scheduler** (can be embedded in controller or standalone) - determines how to partition a model and distribute work across available nodes
3. **Worker Nodes** - execute actual training steps, gather gradients, synchronize parameters

### Key Design Patterns

- **Two-stage orchestration**: Controller handles job lifecycle + global state; Scheduler handles fine-grained task distribution
- **Failed worker recovery**: Workers report health; controller reassigns tasks
- **Parameter Server vs. AllReduce**: Support for both communication modes (configurable per job)
- **Transparent fault tolerance**: Task checkpoints stored centrally; failed workers restart from checkpoint

### Communication Protocols

1. **RPC/gRPC** (controller↔scheduler, scheduler↔nodes): Task scheduling, task status updates
2. **ZMQ** (model runner ↔ network): Fast data transfer during training:
   - `TENSOR` - Large tensor transfers (gradients, activations)
   - `CONTROL` - Control messages (issue step, sync, barrier)
   - `METRIC` - Periodic metrics (loss, throughput, GPU utilization)

### Configuration Strategy

- Use YAML files in `config/` for job, deployment, and environment configurations
- Environment variables override YAML defaults (for secrets, cloud credentials)
- Controller reads `controller.yaml`, Scheduler reads `scheduler.yaml`, Nodes read from a central config map (deprecated) or individual node configs

### Distributed Training Features

- **Automatic scaling**: Add/remove workers dynamically based on workload
- **Gradient synchronization**: Both synchronous (s-SGD) and asynchronous (async-grad) modes
- **Mixed precision**: Support for FP16/FP32 training with loss scaling
- **Pipeline parallelism**: Stop-the-world barrier to share activations across pipeline stages
- **Data parallelism**: Automatic gradient averaging across workers

### Working with Distributed Components

When modifying the distributed system:

1. **Controller changes**: Must ensure thread-safe job queue and worker registration map. Use `concurrent.futures.ThreadPoolExecutor` or `celery` for task queuing.
2. **Scheduler changes**: Focus on task partitioning strategies; ensure schedulers across multiple instances maintain consistent state (shared Redis/etcd often needed).
3. **Network changes**: Prefer async I/O (asyncio) for message receiving; buffer messages during peak loads.
4. **Worker changes**: Minimize blocking operations during training steps; use async communication for heartbeat/status reporting.

### Error Handling Philosophy

- **Fail-fast for users**: Clear error messages when jobs fail (invalid config, vertex bugs, resource exhaustion)
- **Transparent failures**: Workers report failures; controller automatically retries and reassigns tasks
- **No silent retries**: All retry attempts are logged with exponential backoff details

### Testing Distributed Systems

- Tests should use mocking to simulate network partitions, worker failures, and leader elections
- Use `pytest-mock` for mocking RPC calls
- Integration tests should start mock controllers/schedulers and run small training loops
- Consider using `tmpdir` for temporary log/database files during tests

### Python Code Style Requirements

- Follow PEP 8 with `black` formatting
- Type hints for all public functions (use `mypy` strict mode)
- Docstrings for complex functions (use `numpy` style)
- Avoid mutable default arguments
- Use `logging` module (not `print`) for runtime output
- Resource cleanup: Use `contextlib.contextmanager` or try/finally blocks

### Common Gotchas

- **Race conditions in shared state**: Controller's worker registration map can race with scheduler task assignment → use thread-safe data structures
- **ZMQ blocking**: Long messages can block the entire receive loop → use `socket.setsockopt(zmq.REQ_RETRIES, x)`
- **GPU memory leaks**: Model runner must explicitly move tensors back to CPU and delete model references after each step → otherwise workers run out of memory
- **Checkpoint overhead**: Frequent checkpointing slows training throughput → default to 100-500 steps between checkpoints
- **Synchronization barriers**: In pipeline parallelism, ensure all stages participate in barriers; missed barriers cause pipeline bubbles

### Additional Resources

- [Ray Distributed Training](https://docs.ray.io/en/latest/train/)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [Kubernetes Scheduling Principles](https://kubernetes.io/docs/concepts/scheduling-eviction/)