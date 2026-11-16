# GLOSSARY.md

This glossary defines the core concepts and vocabulary for the RMLTP distributed training platform.

- **Job**: A complete training task submitted by a user, consisting of a model configuration, hardware requirements, and training parameters.
- **Task**: An atomic unit of work derived from a Job (e.g., "forward_pass", "backward_pass").
- **Node**: A worker machine that executes Tasks.
- **Worker**: A process running on a Node that communicates with the Controller to pull Tasks.
- **Controller**: The central orchestrator that accepts Jobs, manages the Node registry, and dispatches Tasks to Workers.
- **Scheduler**: The component responsible for partitioning a Job into Tasks and assigning them to specific Nodes.
- **Partition Strategy**: The logic used to split a model or data across multiple Nodes (e.g., Data Parallelism, Pipeline Parallelism).
- **Sync Barrier**: A coordination mechanism that forces all Workers to reach a specific point in training before continuing, used in synchronous training.
- **Checkpoint**: A saved state of the model parameters and optimizer state, used for fault tolerance.
