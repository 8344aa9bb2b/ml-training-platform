# OpenSpec Proposal 001: True Multi-Process Distributed Engine & Cloud Containerization

- **Author**: Platform Team
- **Date**: 2026-11-15
- **Status**: Accepted / Implementation Ready
- **Target Release**: v2.0.0

---

## 1. Context & Motivation

Earlier versions of RMLTP implemented domain models, mathematical synchronization primitives (AllReduce, ParameterServer, Data/Pipeline parallel executors), and unit tests within a unified process. However, to operate as an authentic distributed platform suitable for real multi-terminal demonstrations, assignments, and production deployments:
1. Inter-process communication over ZeroMQ TCP sockets must replace in-memory calls.
2. Worker nodes must run as autonomous command-line processes (`python src/node/main.py`) that register and heartbeat over the network.
3. The platform must be fully containerized (`Dockerfile` and `docker-compose.yml`) and deployable to free cloud hosting (Hugging Face Spaces Docker / Render).

---

## 2. Specification Changes

### 2.1 Networking Specification (`specs/protocols/messaging.md`)
- **Controller Socket**: Transition from `zmq.REQ` to **`zmq.ROUTER`** (or `zmq.REP`) binding on `tcp://0.0.0.0:<port>` to allow concurrent worker connections without deadlocks.
- **Worker Socket**: Connects using **`zmq.DEALER`** (or `zmq.REQ`) to `tcp://<controller_host>:<controller_port>`.
- **Implement Real Connect**: Complete `SocketManager.connect()` with non-blocking polling and async send queues.

### 2.2 Worker Autonomous Daemon (`src/node/worker.py` & `main.py`)
- Complete `Worker.connect_to_controller(address)`:
  - Establishes TCP channel to Controller.
  - Sends `WORKER_REGISTER` payload with PID, CPU cores, GPU info, and memory.
  - Initiates background 2-second heartbeat loop.
  - Enters non-blocking task listener loop awaiting real `TASK_ASSIGNMENT` packets.

### 2.3 Controller Multi-Process Job Dispatcher (`src/controller/controller.py`)
- Complete `Controller.run_job()` and `_workload_scheduler()`:
  - Takes submitted job configurations.
  - Shards real datasets across registered worker processes using `DataParallelPartition`.
  - Sends real task payloads over TCP sockets to workers.
  - Collects results and gradients, synchronizes weights, and writes atomic checkpoints.

### 2.4 Containerization & Free Cloud Deployment Specification
- Provide multi-service `docker-compose.yml` (Controller, Workers, Dashboard).
- Provide single-container multi-process `Dockerfile` with supervisor script (`start_container.sh`) optimized for Hugging Face Spaces (Docker SDK, 16GB free RAM).

---

## 3. Step-by-Step Implementation & Commit Strategy

Every step will be implemented and committed individually with conventional commits:

| Step | Commit Message | Files Affected | Acceptance Criteria |
|---|---|---|---|
| **Step 1** | `feat(network): implement real multi-process ZMQ client-server connection` | `src/network/socket_manager.py`, `tests/unit/test_socket_manager.py` | Two independent sockets communicate over real TCP loopback without deadlocks. |
| **Step 2** | `feat(node): wire autonomous worker CLI daemon and heartbeat handshake` | `src/node/worker.py`, `src/node/main.py`, `tests/unit/test_worker.py` | Running `worker.py` from CLI registers with running controller process over TCP. |
| **Step 3** | `feat(controller): implement multi-process job dispatcher and task execution` | `src/controller/controller.py`, `tests/integration/test_end_to_end.py` | Submitted job shards data and trains across real separate worker processes over TCP. |
| **Step 4** | `feat(resilience): implement live process kill-and-recover demonstration script` | `demo_cluster.py` | Standalone script spawns separate processes, kills one worker, and verifies seamless task recovery. |
| **Step 5** | `feat(docker): add Dockerfile, docker-compose, and cloud deployment scripts` | `Dockerfile`, `docker-compose.yml`, `start_container.sh`, `README.md` | Single command builds container; runs all services and serves dashboard on port 7860/5000. |

---

## 4. Verification & Testing Gate
- `pytest` must pass all test suites (unit, integration, benchmark) before and after each commit.
- Zero regression on existing 111 tests.
- Working multi-terminal command verification.
