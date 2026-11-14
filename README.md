# Distributed ML Training Platform

A distributed ML training platform inspired by Kubernetes + Ray + PyTorch Distributed, providing orchestrator, scheduler, and worker agent components for scalable distributed training.

## Quick Start

```bash
# Install dependencies:
pip install -r requirements.txt

# Start the controller (master orchestrator):
python src/controller/main.py --config config/controller.yaml

# In another terminal, start a worker node:
python src/node/worker.py --config config/nodes.yaml

# Run a training job:
python src/controller/main.py --mode job --job-config config/job.yaml
```

See [CLAUDE.md](CLAUDE.md) for detailed development and architecture information.

## Features

- **Two-stage orchestration**: Controller and Scheduler for job lifecycle and work distribution
- **Automatic scaling**: Dynamic worker registration and task reallocation
- **Distributed training modes**: Synchronous SGD, asynchronous SGD, pipeline parallelism
- **Mixed precision training**: Support for FP16/FP32 training
- **Fault tolerance**: Transparent worker failure recovery and task checkpointing
- **Monitoring**: Prometheus integration for real-time metrics

## Project Structure

```
.
├── src/              # Core application code
│   ├── controller/   # Master orchestrator
│   ├── scheduler/    # Work distribution
│   ├── node/         # Worker agents
│   ├── model/        # Training loops
│   └── network/      # Communication layer
├── core/             # Shared utilities
├── config/           # Configuration files
├── tests/            # Test suite
├── docs/             # Documentation and diagrams
└── logs/             # Runtime logs and checkpoints
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Developer guide
- [docs/architecture.md](docs/architecture.md) - Detailed architecture

## License

MIT