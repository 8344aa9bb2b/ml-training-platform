# OpenSpec: Messaging Protocol Specification

## 1. Protocol Overview

The distributed platform uses a **ZeroMQ (ZMQ)** based message bus for inter-component communication. All messages are serialized as JSON strings for simplicity and compatibility.

### 1.1 Message Structure

```
{
  "message_type": "task_assignment",
  "sender_id": "worker-1",
  "recipient_id": "controller",
  "payload": {
    "task_id": "job-123",
    "task_type": "forward_pass",
    "parameters": {
      "batch_size": 32,
      "device": "cuda:0"
    }
  },
  "message_id": "msg-20231114-123456",
  "timestamp": 1699700000.123
}
```

### 1.2 Message Types

#### 1.2.1 Task Assignment (`task_assignment`)
- **Purpose**: Assigns a training task to a worker.
- **Fields**:
  - `task_id`: Unique identifier for the task
  - `task_type`: Type of operation (forward_pass, backward_pass, optimizer_step, checkpoint)
  - `parameters`: Dictionary of task-specific parameters
  - `priority`: Task priority (integer, higher is more urgent)

#### 1.2.2 Task Status (`task_status`)
- **Purpose**: Worker reports completion status of a task.
- **Fields**:
  - `task_id`: Task identifier
  - `successful`: Boolean indicating success
  - `result`: Execution result (metrics, gradients, etc.)
  - `error`: Error message if task failed

#### 1.2.3 Tensor Data (`tensor_data`)
- **Purpose**: Transfers large tensors (gradients, activations) between components.
- **Fields**:
  - `tensor_name`: Name identifier for the tensor
  - `tensor_data`: Serialized tensor (PyTorch tensor or NumPy array)
  - `shape`: Tensor dimensions
  - `dtype`: Data type (e.g., "float32", "float16")
  - `compression`: Compression method (none, gzip, lz4)

#### 1.2.4 Control Message (`control`)
- **Purpose**: Synchronization and coordination signals.
- **Fields**:
  - `control_type`: Type of control signal (sync_barrier, reset, stop, checkpoint)
  - `parameters`: Additional parameters for the control operation
  - `expected_barriers`: Number of expected participants for barriers
  - `barrier_id`: Unique barrier identifier

#### 1.2.5 Metric Message (`metric`)
- **Purpose**: Periodic reporting of training metrics.
- **Fields**:
  - `metrics`: Dictionary of metric names and values
  - `step`: Current training step
  - `epoch`: Current training epoch
  - `worker_id`: Worker identifier

#### 1.2.6 Heartbeat (`heartbeat`)
- **Purpose**: Worker health check.
- **Fields**:
  - `sender_id`: Worker identifier
  - `timestamp`: Current time

### 1.3 Serialization

- **Format**: JSON (human-readable and machine-parsable)
- **Binary**: For production, consider Protobuf for performance.
- **Versioning**: Messages support versioning (`v1`, `v2`) to enable backward compatibility.

## 2. Protocol Implementation Details

### 2.1 ZeroMQ Topology

- **Controller ↔ Scheduler**: Publish-Subscribe (pubsub) pattern for task distribution.
- **Controller ↔ Worker**: Request-Reply (reqrep) pattern for task assignment and status updates.
- **Worker ↔ Worker**: Direct peer-to-peer for gradient synchronization (optional).

### 2.2 Message Flow

1. **Job Submission**:
   - Client submits job to Controller via REST API.
   - Controller validates job and creates job ID.

2. **Worker Registration**:
   - Worker connects to Controller and sends registration message.
   - Controller stores worker capabilities and health status.

3. **Task Assignment**:
   - Scheduler selects appropriate worker based on resource availability.
   - Controller sends `task_assignment` message to worker.

4. **Task Execution**:
   - Worker executes task and reports status via `task_status` message.

5. **Metrics Reporting**:
   - Worker sends periodic `metric` messages to Controller.

6. **Checkpointing**:
   - Worker sends `checkpoint` message when saving model state.

7. **Heartbeat**:
   - Worker sends periodic `heartbeat` messages to keep alive.

## 3. Versioning Strategy

- **Major Version**: Breaking changes (e.g., message format, protocol version).
- **Minor Version**: Non-breaking additions (new message types).
- **Patch Version**: Bug fixes and minor improvements.

Current protocol version: **v1.0.0**

## 4. Security Considerations

- **Authentication**: Workers use unique identifiers and optional TLS for secure connections.
- **Authorization**: Controller validates worker credentials before accepting messages.
- **Data Integrity**: Use message checksums for data validation.
- **Rate Limiting**: Controller implements rate limiting on registration and task assignment.

## 5. Future Extensions

- **Protocol Buffers**: Replace JSON with Protobuf for performance.
- **TLS Encryption**: Add TLS support for secure communication.
- **Reliable Messaging**: Implement message acknowledgment and retransmission.
- **Compression**: Add optional compression for large tensor transfers.