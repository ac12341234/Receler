# AGENTS.md

## Project Context

Receler is the official implementation of "Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers".
The current code trains one Eraser per concept with `train_receler.py` and runs inference with `test_receler.py` by loading one model folder containing:

- `eraser_weights.pt`
- `eraser_config.json`
- optional visualization/checkpoint outputs

The planned extension is multi-Eraser fusion for inference and learnable fusion weights.

## Current Design Decisions

- Implement multi-Eraser fusion on the Diffusers path only.
- Do not convert saved Diffusers-format Eraser checkpoints back to LDM format.
- Do not add `MultiBasicTransformerBlockWithEraser` or replace Diffusers `BasicTransformerBlock` objects.
- Use `register_forward_hook` on Diffusers cross-attention modules to apply fusion.
- Fusion formula for an attention output is:

```python
output + sum(active_weight_i * eraser_i(output))
```

- Different Erasers may cover different layer sets. Use the union of all checkpoint layer names.
- For each layer, only Erasers that have that layer are active.
- Missing Erasers contribute zero at that layer.
- Normalize weights with softmax over active Erasers only.

## Files To Touch For Multi-Eraser Fusion

Keep the implementation scoped to these files unless there is a strong reason to expand:

- `receler/multi_eraser.py`
  - New `MultiEraserWrapper`.
  - Load multiple Eraser folders.
  - Build per-layer adapter modules from `eraser_weights.pt`.
  - Register and remove hooks.
  - Manage fusion weights/logits.
  - Export `fusion_config.json` data.

- `test_receler.py`
  - Keep existing `--model_name_or_path` behavior.
  - Add `--eraser_paths`, `--fusion_weight`, and `--fusion_config`.
  - In multi-Eraser mode, load the base `CompVis/stable-diffusion-v1-4` pipeline and attach `MultiEraserWrapper`.

- `train_fusion.py`
  - New Diffusers-based trainer for fusion weights.
  - Freeze Stable Diffusion and all Eraser adapter parameters.
  - Train only fusion logits.
  - Save learned normalized weights to `fusion_config.json`.

## Compatibility Requirements

- Existing single-Eraser commands must continue to work.
- Existing saved Eraser folders remain valid inputs.
- `--fusion_config` and `--eraser_paths` should be mutually exclusive.
- If `--fusion_weight` is omitted, initialize equal weights.
- If `--fusion_weight` is provided, its length must match `--eraser_paths`.
- Learned fusion configs should be usable directly by `test_receler.py`.

## Testing Notes

At minimum, verify:

- Single-Eraser inference path remains unchanged.
- Multi-Eraser CLI parsing handles path and weight mismatches clearly.
- `fusion_config.json` loads the same paths and weights used for inference.
- Hooks are removed cleanly when requested.
- Only fusion logits require gradients in `train_fusion.py`.
