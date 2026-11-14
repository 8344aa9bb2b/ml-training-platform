# OpenSpec: Architecture Overview

## 1. System Topology

```
                  +-------------------+
                  |    Controller     |
                  |  (Job Orchestrator) |
                  +--------+----------+
                           |
            +--------------+--------------+
            |                             |
    +-------v-------+             +-------v-------+
    |   Worker 1    |             |   Worker 2    |
    | (Node Agent)  |             | (Node Agent)  |
    +---------------+             +---------------+
```

## 2. Core Components

### 2.1 Controller (Master Node)
- **Role**: Coordinates overall cluster state, manages resources, assigns jobs.
- **State**: In-memory worker map, persistent job execution database (optional SQLite/Redis).
- **Interface**: Exposes gRPC/REST endpoints for client jobs and HTTP endpoints for Prometheus metrics.

### 2.2 Scheduler (Work Distribution)
- **Role**: Decides task placement based on node capability (CPU, GPU, Memory).
- **Logic**: Implements round-robin, least-loaded, or locality-aware placement algorithms.

### 2.3 Worker Nodes
- **Role**: Executes actual forward/backward passes and parameter updates.
- **Sub-components**:
  - `SocketManager`: Handles non-blocking ZMQ messaging.
  - `ModelRunner`: Executes PyTorch operations.
  - `ResourceTracker`: Reports GPU/CPU stats.

## 3. Communication Patterns

- **Control Loop**: Push-Pull or Pub-Sub via ZeroMQ for fast non-blocking status updates.
- **Data Transfers**: Heavy tensor data sent using direct ZMQ sockets or shared memory (when co-located).
- **Heartbeat**: 5-second interval heartbeat from Worker to Controller. Missing 3 consecutive heartbeats triggers a `NodeDead` event.
