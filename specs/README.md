# OpenSpec: Distributed ML Training Platform Specs

This directory contains the formal specifications for the RMLTP (Ray-based ML Training Platform).

## Structure

```
specs/
├── architecture/     # System architecture and data flow specs
│   └── overview.md   # High-level system design
├── protocols/        # Communication protocols and serialization specs
│   └── messaging.md  # ZMQ/gRPC message formats
└── api/              # REST & gRPC API specifications
    └── controller.md # Controller endpoints and contracts
```

## Specification Rules

1. **Protocol Definitions**: All message schemas must define field names, types, and binary representation.
2. **Backward Compatibility**: Protocols use versioning (`v1`, `v2`) to ensure rolling updates don't break workers.
3. **Idempotency**: All control signals (start, stop, reset) must be idempotent.
