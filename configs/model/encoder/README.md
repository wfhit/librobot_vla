# Encoder Configuration Directory

This directory contains configuration files for proprioceptive state encoders used in the VLA framework.

## Purpose

Encoders process proprioceptive observations (joint positions, velocities, forces, etc.) into features that are fused with vision-language features.

## Supported Encoders

- **MLP**: Multi-layer perceptron encoder
- **Transformer**: Temporal transformer for sequence encoding
- **RNN/LSTM**: Recurrent encoders for temporal information
- **Identity**: Pass-through encoder (no encoding)

## Configuration Structure

Each encoder config should include:

```yaml
# Encoder type (must match registry name)
type: "mlp"

# Input configuration
input_dim: 14  # Dimension of proprioceptive state
normalize_input: true

# Architecture parameters
hidden_dims: [256, 128]
output_dim: 128
activation: "relu"
dropout: 0.1

# Optional: temporal encoding
use_temporal: false
sequence_length: 10
```

## Usage

Reference encoder configs in experiment configs:
```yaml
model:
  encoder: "configs/model/encoder/mlp.yaml"
```
